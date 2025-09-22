import os
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchaudio.functional as F # for edit distance
from pathlib import Path
from omegaconf import OmegaConf
from functools import partial
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


import pdb


from src.dataset.utils import train_test_split_indices, get_all_files
from src.dataset.dataset import BrainToTextDataset
from src.train.utils import transform_data
from src.dataset.utils import get_all_files
from src.inference.utils import calculate_per


from src.train.models import GRUDecoder

from src.utils import log_info

class Trainer:
    def __init__(self, config, logger, model):
        log_info(
            logger=logger,
            text="Initializing Traiiner"
        )
        self.config = config
        self.logger = logger
        self.model = model

        self.best_val_loss = torch.inf
        self.best_val_per = torch.inf
        self.cur_val_per = torch.inf
        self.cur_val_loss = torch.inf
    
        self._setup_conf_params()
        self._device_setup()
        self._log_model_info()
        self._dir_setup()

        self.optimizer = self.create_optimizer()
        if self.lr_scheduler_type == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.lr_min/self.lr_max,
                total_iters=self.lr_decay_steps
            )

        else:
            self.lr_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        self.CTC_loss = torch.nn.CTCLoss(
            blank=0,
            reduction='none',
            zero_infinity=False
        )

        if self.init_from_checkpoint:
            self.load_model_checkpoint()

    def _setup_conf_params(self):
        self.logger.info('Setting training config parameters')
        train_config = self.config['train']
        self.device = train_config['device']
        seed = self.config['dataset']['info']['seed']
        
        self.n_train_batches =  train_config['n_train_batches'] 
        self.lr_scheduler_type = train_config['lr_scheduler_type']
        self.lr_max = train_config['lr_max']
        self.lr_min = train_config['lr_min']
        self.lr_decay_steps = train_config['lr_decay_steps']
        self.lr_warmup_steps = train_config['lr_warmup_steps']
        self.lr_max_day = train_config['lr_max_day']
        self.lr_min_day = train_config['lr_min_day']
        self.lr_decay_steps_day = train_config['lr_decay_steps_day']
        self.lr_warmup_steps_day = train_config['lr_warmup_steps_day']
        
        self.weight_decay_day =  train_config['weight_decay_day'] # weight decay for the day specific input layers
        self.beta0 = train_config['beta0']
        self.beta1 = train_config['beta1']
        self.epsilon = train_config['epsilon']
        self.weight_decay = train_config['weight_decay']
        self.weight_decay_day = train_config['weight_decay_day']
        self.seed = train_config['seed']
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
        pass

    def _device_setup(self):
        """Setup device and move model to device"""
        self.logger.info("Setting up device")

        if self.device == 'cpu':
            self.device = torch.device('cpu')
            return

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using device: {self.device}")
            if torch.cuda.device_count() > 1:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device('cpu')
            self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

    def _log_model_info(self):
        self.logger.info('Checking model parameters')
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")
        self.logger.info(self.model)

    def _dir_setup(self):
        self.logger.info("Setting directories")
        dir_conf = self.config['directory']
        output_dir = dir_conf['output_dir']
        checkpoint_dir = dir_conf['checkpoint_dir']

        cur_dir = os.getcwd()
        self.output_dir = Path(cur_dir, output_dir)
        self.checkpoint_dir = Path(cur_dir, checkpoint_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Outputs dir:: {self.output_dir}")
        self.logger.info(f"Checkpoints dir:: {self.checkpoint_dir}")

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        self.logger.info("Creating optimizer")
        bias_params = [p for name, p in self.model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.lr_max_day, 'weight_decay' : self.weight_decay_day, 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.lr_max,
            betas = (self.beta0, self.beta1),
            eps = self.epsilon,
            weight_decay = self.weight_decay,
            fused = True
        )

        return optim 
    
    def create_cosine_lr_scheduler(self, optim):
        """
        Creates a cosine learning rate scheduler with warmup.
        Supports different LR schedules for different parameter groups.
        """
        self.logger.info("Creating cosine LR scheduler")

        def lr_lambda(step: int, min_lr_ratio: float, decay_steps: int, warmup_steps: int) -> float:
            """Cosine decay with linear warmup."""
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            elif step < decay_steps:
                progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            return min_lr_ratio

        # Map param groups to their LR settings
        param_group_configs = []
        if len(optim.param_groups) == 3:
            param_group_configs = [
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),       # biases
                (self.lr_min_day / self.lr_max_day, self.lr_decay_steps_day, self.lr_warmup_steps_day),  # day params
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),       # rest of weights
            ]
        elif len(optim.param_groups) == 2:
            param_group_configs = [
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),       # biases
                (self.lr_min / self.lr_max, self.lr_decay_steps, self.lr_warmup_steps),       # rest of weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")

        # Create LambdaLR schedulers using functools.partial to avoid repeated lambdas
        lr_lambdas = [partial(lr_lambda, min_lr_ratio=cfg[0], decay_steps=cfg[1], warmup_steps=cfg[2])
                    for cfg in param_group_configs]

        return LambdaLR(optim, lr_lambdas)

    def load_model_checkpoint(self):
        self.logger.info("Loading model checkpoint")
        self.logger.info(f"Path :: {self.init_checkpoint_path}")

        checkpoint = torch.load(self.init_checkpoint_path, weights_only = False) # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_per'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)
        
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model checkpoint")

    def save_model_checkpoint(self, batch_no, best=False):
        self.logger.info("Saving model checkpoint")
        
        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.lr_scheduler.state_dict(),
            'val_per' : self.cur_val_per,
            'val_loss' : self.cur_val_loss
        }
        if not best:
            path = Path(self.checkpoint_dir, f'checkpoint-{batch_no}')
            torch.save(checkpoint, path)
            self.logger.info("Saved model to checkpoint: " + path)
        else:
            path = Path(self.checkpoint_dir, f'best_model')
            torch.save(checkpoint, path)
            self.logger.info(f"Saved model to checkpoint: {path}")
        
        with open(os.path.join(self.checkpoint_dir, 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.config, f=f)

    def _get_batch(self, batch):
        features = batch['input_features'].to(self.device)
        labels = batch['seq_class_ids'].to(self.device)
        n_time_steps = batch['n_time_steps'].to(self.device)
        phone_seq_lens = batch['phone_seq_lens'].to(self.device)
        day_indicies = batch['day_indices'].to(self.device)

        return features, labels, n_time_steps, phone_seq_lens, day_indicies

    def _train_batch(self, batch):
        self.model.train()
        features, labels, n_time_steps, phone_seq_lens, day_indicies = self._get_batch(batch)
        features, n_time_steps = transform_data(
            config=self.config,
            features=features,
            n_time_steps=n_time_steps,
            device=self.device,
            mode='train'
        )

        adjusted_lens = ((n_time_steps - self.model_patch_size) / self.model_patch_stride + 1).to(torch.int32)

        logits = self.model(features, day_indicies)

        loss = self.CTC_loss(
            log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
            targets = labels,
            input_lengths = adjusted_lens,
            target_lengths = phone_seq_lens
        )

        return loss

    def train(self, train_loader, val_loader):
        self.logger.info("Starting training")
        
        trains_losses = []
        val_losses = []
        val_results = []

        train_pers = []
        val_pers = []

        val_steps_since_improvemnt = 0
        train_start_time = time.time()

        for i , batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            start_time = time.time()

            
            loss = self._train_batch(batch=batch)
            loss = torch.mean(loss)

            loss.backward()

            if self.grad_norm_clip_value > 0: 
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm = self.grad_norm_clip_value,
                    error_if_nonfinite = True,
                    foreach = True
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            train_step_duration = time.time() - start_time
            trains_losses.append(loss.detach().item())

            if i % self.batches_per_train_log == 0:
                self.logger.info(f'Train batch {i}: ' +
                    f'loss: {(loss.detach().item()):.2f} ' +
                    f'grad norm: {grad_norm:.2f} '
                    f'time: {train_step_duration:.3f}')
            
            if i % self.batches_per_val_step == 0:
                self.logger.info(f"Running val after {i} batches")
                start_time = time.time()
                val_metrics, session_day_dict = self.validation(
                    val_loader=val_loader
                )
                val_step_duration = time.time() - start_time
                self.logger.info(f'Validation batch {i}: ' +
                        f'PER (avg): {val_metrics["avg_per"]:.4f} ' +
                        f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                        f'time: {val_step_duration:.3f}')
                
                if self.log_per_by_day:
                    for day in val_metrics['day_pers'].keys():
                        self.logger.info(
                            f"Day:{day} Session:{session_day_dict[day]} Val PER:{val_metrics['day_pers'][day]:0.4f}"
                        )
                    
                self.cur_val_per = val_metrics['avg_per']
                self.cur_val_loss = val_metrics['avg_loss']
                val_pers.append(self.cur_val_per)
                val_losses.append(self.cur_val_loss)
                val_results.append(val_metrics)

                if self.cur_val_loss< self.best_val_per:
                    self.save_model_checkpoint(batch_no=i, best=True)
                    val_steps_since_improvemnt = 0
                else:
                    val_steps_since_improvemnt += 1
                
            if i+1 % self.checkpoint_save_interval == 0:
                self.save_model_checkpoint(batch_no=i, best=False)
            
            if val_steps_since_improvemnt>20:
                self.logger.info(" EARLY STOPPED")
                break

        train_duration = time.time()-train_start_time
        self.logger.info(f"Best avg val PER {self.best_val_PER:.5f}")
        self.logger.info(f'Total training time: {(train_duration / 60):.2f} minutes')

    def validation(self, val_loader) -> Tuple[Dict[str, Any], Dict[int, str]]:
        """
        Run validation loop with CTC loss and PER calculation using logits directly.

        Args:
            val_loader: PyTorch DataLoader for validation data.

        Returns:
            metrics (dict): Aggregated validation metrics including avg_loss and PER.
            session_day_dict (dict): Mapping from day index to session name.
        """
        self.model.eval()

        metrics = {
            "logits": [],
            "true_seq": [],
            "phone_seq_lens": [],
            "transcription": [],
            "losses": [],
            "block_nums": [],
            "trial_nums": [],
            "day_indices": [],
        }

        # Collect sessions per day
        filepaths = get_all_files(
            parent_dir=self.config["dataset"]['info']["dataset_dir"], kind="val"
        )
        day_index = 0
        session_day_dict = {}
        for session, paths in filepaths.items():
            if not session.startswith("t15.20") or len(paths) < 1:
                day_index += 1
                continue
            session_day_dict[day_index] = session
            day_index += 1

        all_logits = []
        all_true = []
        all_lens = []
        all_day_indices = []

        for batch in val_loader:
            with torch.no_grad():
                features, labels, n_time_steps, phone_seq_lens, day_indices = self._get_batch(batch)
                adjusted_lens = ((n_time_steps - self.model_patch_size) / self.model_patch_stride + 1).to(torch.int32)
                logits = self.model(features, day_indices)

                loss = self.CTC_loss(
                    log_probs=logits.log_softmax(2).permute(1, 0, 2),
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens,
                ).mean()

            metrics["losses"].append(loss.item())

            # Store logits and other info for PER calculation
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

        # Compute PER using logits directly
        avg_per, day_per = calculate_per(all_logits, all_true, all_lens, all_day_indices)
        metrics["avg_per"] = avg_per
        metrics["day_pers"] = day_per
        metrics["avg_loss"] = float(np.mean(metrics["losses"]))

        return metrics, session_day_dict


    def validation1(self, val_loader) -> Tuple[Dict[str, Any], Dict[int, str]]:
        """
        Run validation loop with CTC loss and phone error rate (PER) calculation.

        Args:
            val_loader: PyTorch DataLoader for validation data.

        Returns:
            metrics (dict): Aggregated validation metrics.
            session_day_dict (dict): Mapping from day index to session name.
        """
        self.model.eval()

        metrics = {
            "decoded_seqs": [],
            "true_seq": [],
            "phone_seq_lens": [],
            "transcription": [],
            "losses": [],
            "block_nums": [],
            "trial_nums": [],
            "day_indices": [],
        }

        total_edit_distance = 0
        total_seq_length = 0
        day_per = {}

        # Collect sessions per day
        filepaths = get_all_files(
            parent_dir=self.config["dataset"]['info']["dataset_dir"], kind="val"
        )

        day_index = 0
        session_day_dict = {}
        for session, paths in filepaths.items():
            if not session.startswith("t15.20") :
                continue
            if len(paths)<1:
                day_index += 1
                continue
            day_per[day_index] = {"total_edit_distance": 0, "total_seq_length": 0}
            session_day_dict[day_index] = session
            day_index += 1

        # Validation loop
        for batch in val_loader:
            with torch.no_grad():
                features, labels, n_time_steps, phone_seq_lens, day_indicies = (
                    self._get_batch(batch)
                )
                adjusted_lens = (
                    (n_time_steps - self.model_patch_size) / self.model_patch_stride + 1
                ).to(torch.int32)

                logits = self.model(features, day_indicies)

                loss = self.CTC_loss(
                    log_probs=logits.log_softmax(2).permute(1, 0, 2),
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens,
                ).mean()

            metrics["losses"].append(loss.item())

            # Decode predictions and compute edit distance
            batch_edit_distance = 0
            decoded_seqs = []

            for idx in range(logits.shape[0]):
                decoded_seq = torch.argmax(
                    logits[idx, : adjusted_lens[idx], :], dim=-1
                )
                decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                decoded_seq = decoded_seq.cpu().numpy()
                decoded_seq = np.array([i for i in decoded_seq if i != 0])

                true_seq = labels[idx, : phone_seq_lens[idx]].cpu().numpy()

                batch_edit_distance += F.edit_distance(decoded_seq, true_seq)
                decoded_seqs.append(decoded_seq)

            # Day-level aggregation
            day = batch["day_indices"][0].item()
            if day in day_per:
                day_per[day]["total_edit_distance"] += batch_edit_distance
                day_per[day]["total_seq_length"] += phone_seq_lens.sum().item()

            total_edit_distance += batch_edit_distance
            total_seq_length += phone_seq_lens.sum().item()

            # Save batch metrics
            metrics["decoded_seqs"].append(decoded_seqs)
            metrics["true_seq"].append(batch["seq_class_ids"].cpu().numpy())
            metrics["phone_seq_lens"].append(batch["phone_seq_lens"].cpu().numpy())
            metrics["transcription"].append(batch["transcriptions"])
            metrics["block_nums"].append(batch["block_nums"].numpy())
            metrics["trial_nums"].append(batch["trial_nums"].numpy())
            metrics["day_indices"].append(batch["day_indices"].cpu().numpy())

        # Aggregate metrics
        avg_per = total_edit_distance / total_seq_length
        metrics["day_pers"] = day_per
        metrics["avg_per"] = avg_per
        metrics["avg_loss"] = float(np.mean(metrics["losses"]))

        return metrics, session_day_dict



def training_pipeline(config, logger):
    train_conf = config['train']
    pin_memory = train_conf['pin_memory']
    shuffle = train_conf['shuffle_data']
    n_data_workers = train_conf['n_data_workers']

    
    train_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='train', logger=logger)
    train_trials, _ = train_test_split_indices(
        config=config,
        file_paths=train_filepaths,
        logger=logger
    )
    train_ds = BrainToTextDataset(
        trial_indices=train_trials, config=config,
        logger=logger, kind='train'
    )
    train_loader = DataLoader(
        train_ds, batch_size=None,  shuffle=shuffle,
        num_workers=n_data_workers, pin_memory=pin_memory
    )

    val_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='val', logger=logger)
    val_trails, _ = train_test_split_indices(
        config=config,
        file_paths=val_filepaths,
        logger=logger
    )
    val_ds = BrainToTextDataset(
        trial_indices=val_trails, config=config,
        logger=logger, kind='test'
    )
    val_loader =  DataLoader(
        val_ds, batch_size=None, shuffle=False,
        num_workers=n_data_workers, pin_memory=pin_memory
    )
    

    model = GRUDecoder(
        config=config
    )
    
    trainer = Trainer(
        config=config,
        logger=logger,
        model=model
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

