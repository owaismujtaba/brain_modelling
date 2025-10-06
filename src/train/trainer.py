import os
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from omegaconf import OmegaConf
from functools import partial
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset.utils import train_test_split_indices, get_all_files
from src.dataset.dataset import BrainToTextDataset
from src.train.utils import transform_data
from src.inference.utils import calculate_per
from src.train.models import GRUDecoder
from src.utils import log_info

import pdb

class Trainer:
    def __init__(self, config, logger, model):
        log_info(logger, "Initializing Trainer")
        self.config = config
        self.logger = logger
        self.model = model
        self.scaler = GradScaler()
        self.best_val_loss = torch.inf
        self.best_val_per = torch.inf
        self.cur_val_loss = torch.inf
        self.cur_val_per = torch.inf

        self._setup_conf_params()
        self._device_setup()
        self._log_model_info()
        self._dir_setup()

        self.optimizer = self.create_optimizer()
        if self.lr_scheduler_type == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.lr_min / self.lr_max,
                total_iters=self.lr_decay_steps
            )
        else:
            self.lr_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        self.CTC_loss = nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)

        if self.init_from_checkpoint:
            self.load_model_checkpoint()

    def _setup_conf_params(self):
        self.logger.info('Setting training config parameters')
        train_config = self.config['train']
        self.device = train_config['device']
        seed = self.config['dataset']['info']['seed']

        self.n_train_batches = train_config['n_train_batches']
        self.lr_scheduler_type = train_config['lr_scheduler_type']
        self.lr_max = train_config['lr_max']
        self.lr_min = train_config['lr_min']
        self.lr_decay_steps = train_config['lr_decay_steps']
        self.lr_warmup_steps = train_config['lr_warmup_steps']
        self.lr_max_day = train_config['lr_max_day']
        self.lr_min_day = train_config['lr_min_day']
        self.lr_decay_steps_day = train_config['lr_decay_steps_day']
        self.lr_warmup_steps_day = train_config['lr_warmup_steps_day']

        self.weight_decay = train_config['weight_decay']
        self.weight_decay_day = train_config['weight_decay_day']
        self.beta0 = train_config['beta0']
        self.beta1 = train_config['beta1']
        self.epsilon = train_config['epsilon']
        self.grad_norm_clip_value = train_config['grad_norm_clip_value']
        self.batches_per_train_log = train_config['batches_per_train_log']
        self.batches_per_val_step = train_config['batches_per_val_step']
        self.init_from_checkpoint = train_config['init_from_checkpoint']
        self.init_checkpoint_path = train_config['init_checkpoint_path']
        self.model_patch_size = self.config['model']['patch_size']
        self.model_patch_stride = self.config['model']['patch_stride']
        self.log_per_by_day = self.config['logging']['log_per_by_day']
        self.checkpoint_save_interval = train_config['checkpoint_save_interval']

        if seed != -1:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _device_setup(self):
        self.logger.info("Setting up device")
        if self.device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

    def _log_model_info(self):
        self.logger.info('Model parameter summary')
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        day_params = sum(p.numel() for name, p in self.model.named_parameters() if 'day' in name)
        self.logger.info(f"Day-specific parameters: {day_params:,} ({(day_params / total_params * 100):.2f}%)")
        self.logger.info(self.model)

    def _dir_setup(self):
        self.logger.info("Setting up directories")
        dir_conf = self.config['directory']
        cur_dir = os.getcwd()
        self.output_dir = Path(cur_dir, dir_conf['output_dir'])
        self.checkpoint_dir = Path(cur_dir, dir_conf['checkpoint_dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

    def create_optimizer(self):
        self.logger.info("Creating optimizer")
        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        param_groups = [{'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'}]
        if len(day_params) > 0:
            param_groups.append({'params': day_params, 'lr': self.lr_max_day, 'weight_decay': self.weight_decay_day, 'group_type': 'day_layer'})
        param_groups.append({'params': other_params, 'group_type': 'other'})

        optim = torch.optim.AdamW(
            param_groups,
            lr=self.lr_max,
            betas=(self.beta0, self.beta1),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            fused=True
        )
        return optim

    def create_cosine_lr_scheduler(self, optim):
        self.logger.info("Creating cosine LR scheduler with warmup")

        def lr_lambda(step, min_lr_ratio, decay_steps, warmup_steps):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            elif step < decay_steps:
                progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
            return min_lr_ratio

        param_group_configs = []
        if len(optim.param_groups) == 3:
            param_group_configs = [
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),
                (self.lr_min_day / self.lr_max_day, self.lr_decay_steps_day, self.lr_warmup_steps_day),
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),
            ]
        elif len(optim.param_groups) == 2:
            param_group_configs = [
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),
            ]
        else:
            raise ValueError(f"Invalid number of param groups: {len(optim.param_groups)}")

        lr_lambdas = [partial(lr_lambda, min_lr_ratio=cfg[0], decay_steps=cfg[1], warmup_steps=cfg[2])
                      for cfg in param_group_configs]
        return LambdaLR(optim, lr_lambdas)

    def load_model_checkpoint(self):
        self.logger.info(f"Loading checkpoint: {self.init_checkpoint_path}")
        checkpoint = torch.load(self.init_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_per = checkpoint.get('val_per', torch.inf)
        self.best_val_loss = checkpoint.get('val_loss', torch.inf)
        self.model.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def save_model_checkpoint(self, batch_no, best=False):
        path = Path(
            self.checkpoint_dir,
            'best_model' if best else f'checkpoint-{batch_no}_loss-{self.cur_val_loss:.4f}_per-{self.cur_val_per:.4f}'
        )

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'val_per': self.cur_val_per,
            'val_loss': self.cur_val_loss
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        OmegaConf.save(config=self.config, f=Path(self.checkpoint_dir, 'args.yaml'))

    def _get_batch(self, batch):
        features = batch['input_features'].to(self.device)
        labels = batch['seq_class_ids'].to(self.device)
        n_time_steps = batch['n_time_steps'].to(self.device)
        phone_seq_lens = batch['phone_seq_lens'].to(self.device)
        day_indices = batch['day_indices'].to(self.device)
        return features, labels, n_time_steps, phone_seq_lens, day_indices

    def _train_batch(self, batch):
        self.model.train()
        features, labels, n_time_steps, phone_seq_lens, day_indices = self._get_batch(batch)
        features, n_time_steps = transform_data(self.config, features, n_time_steps, self.device, mode='train')
        adjusted_lens = ((n_time_steps - self.model_patch_size) / self.model_patch_stride + 1).to(torch.int32)

        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_str, dtype=torch.float16):
            logits = self.model(features, day_indices)
            loss = self.CTC_loss(
                log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
                targets=labels,
                input_lengths=adjusted_lens,
                target_lengths=phone_seq_lens
            ).mean()

        return loss, logits

    def train(self, train_loader, val_loader):
        self.logger.info("Starting training")
        trains_losses, val_losses, val_results, val_pers = [], [], [], []
        val_steps_since_improvement = 0
        train_start_time = time.time()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100)
        for i, batch in progress_bar:
            self.optimizer.zero_grad()
            start_time = time.time()

            loss, _ = self._train_batch(batch)
            if not torch.isfinite(loss):
                self.logger.warning(f"Non-finite loss detected: {loss.item()}. Skipping step.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_norm_clip_value > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_norm_clip_value,
                    error_if_nonfinite=False,
                    foreach=True
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            trains_losses.append(loss.detach().item())
            train_step_duration = time.time() - start_time

            if i % self.batches_per_train_log == 0:
                self.logger.info(f'Train batch {i}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}, time={train_step_duration:.3f}s')

            if i % self.batches_per_val_step == 0 and i > 0:
                self.logger.info(f"Running validation after {i} batches")
                val_metrics, session_day_dict = self.validation(val_loader)
                self.cur_val_per = val_metrics['avg_per']
                self.cur_val_loss = val_metrics['avg_loss']

                val_pers.append(self.cur_val_per)
                val_losses.append(self.cur_val_loss)
                val_results.append(val_metrics)

                if self.log_per_by_day:
                    for day, per in val_metrics['day_pers'].items():
                        self.logger.info(f"Day {day} | Session {session_day_dict[day]} | Val PER: {per:.4f}")

                if self.cur_val_per < self.best_val_per:
                    self.logger.info(f"New best PER: {self.cur_val_per:.4f} (prev {self.best_val_per:.4f}), saving model.") 
                    self.best_val_per = self.cur_val_per
                    self.best_val_loss = self.cur_val_loss
                    self.save_model_checkpoint(batch_no=i, best=True)
                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement += 1

            if (i+1) % self.checkpoint_save_interval == 0:
                self.save_model_checkpoint(batch_no=i, best=False)

            if val_steps_since_improvement > 20:
                self.logger.info("Early stopping triggered")
                break

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "grad_norm": f"{grad_norm:.4f}" if self.grad_norm_clip_value > 0 else "N/A"
            })

        total_duration = time.time() - train_start_time
        self.logger.info(f"Best avg val PER: {self.best_val_per:.5f}")
        self.logger.info(f"Total training time: {total_duration/60:.2f} minutes")

    def validation(self, val_loader):
        self.model.eval()
        metrics = {
            "logits": [], "true_seq": [], "phone_seq_lens": [], "transcription": [],
            "losses": [], "block_nums": [], "trial_nums": [], "day_indices": []
        }

        filepaths = get_all_files(self.config["dataset"]['info']["dataset_dir"], kind="val")
        session_day_dict = {idx: s for idx, s in enumerate(filepaths.keys())}

        all_logits, all_true, all_lens, all_day_indices = [], [], [], []

        for batch in val_loader:
            with torch.no_grad():
                features, labels, n_time_steps, phone_seq_lens, day_indices = self._get_batch(batch)
                adjusted_lens = ((n_time_steps - self.model_patch_size) / self.model_patch_stride + 1).to(torch.int32)
                logits = self.model(features, day_indices)
                loss = self.CTC_loss(
                    log_probs=logits.log_softmax(2).permute(1,0,2),
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens
                ).mean()

            metrics["losses"].append(loss.item())
            all_logits.append(logits)
            all_true.append(labels)
            all_lens.extend(phone_seq_lens.tolist())
            all_day_indices.extend(day_indices.tolist())

            metrics["true_seq"].append(batch["seq_class_ids"].cpu().numpy())
            metrics["phone_seq_lens"].append(batch["phone_seq_lens"].cpu().numpy())
            metrics["transcription"].append(batch["transcriptions"])
            metrics["block_nums"].append(batch["block_nums"].numpy())
            metrics["trial_nums"].append(batch["trial_nums"].numpy())
            metrics["day_indices"].append(batch["day_indices"].cpu().numpy())

        avg_per, day_per = calculate_per(all_logits, all_true, all_lens, all_day_indices)
        metrics["avg_per"] = avg_per
        metrics["day_pers"] = day_per
        metrics["avg_loss"] = float(np.mean(metrics["losses"]))

        return metrics, session_day_dict


def training_pipeline(config, logger):
    train_conf = config['train']
    pin_memory = train_conf['pin_memory']
    shuffle = train_conf['shuffle_data']
    n_data_workers = train_conf['n_data_workers']

    train_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='train', logger=logger)

    train_trials, _ = train_test_split_indices(config=config, file_paths=train_filepaths, logger=logger)
    train_ds = BrainToTextDataset(trial_indices=train_trials, config=config, logger=logger, kind='train')
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=shuffle, num_workers=n_data_workers, pin_memory=pin_memory)

    val_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='val', logger=logger)
    val_trials, _ = train_test_split_indices(config=config, file_paths=val_filepaths, logger=logger)
    val_ds = BrainToTextDataset(trial_indices=val_trials, config=config, logger=logger, kind='test')
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=n_data_workers, pin_memory=pin_memory)

    model = GRUDecoder(config=config)
    trainer = Trainer(config=config, logger=logger, model=model)
    trainer.train(train_loader=train_loader, val_loader=val_loader)
