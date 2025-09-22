import os
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from omegaconf import OmegaConf

from src.dataset.utils import train_test_split_indices, get_all_files
from src.dataset.dataset import BrainToTextDataset
from torch.utils.data import DataLoader
from src.train.utils import transform_data


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
        self.cur_val_loss = torch.inf
        self.cur_val_per = torch.inf

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
        self.logger.info("Creating cosine LR schedular")
        lr_max = self.lr_max
        lr_min = self.lr_min
        lr_decay_steps = self.lr_decay_steps
        lr_max_day =  self.lr_max_day
        lr_min_day = self.lr_min_day
        lr_decay_steps_day = self.lr_decay_steps_day
        lr_warmup_steps = self.lr_warmup_steps
        lr_warmup_steps_day = self.lr_decay_steps_day

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)

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

    def save_model_checkpoint(self, path):
        self.logger.info("Saving model checkpoint")

        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.lr_scheduler.state_dict(),
            'val_per' : self.cur_val_per,
            'val_loss' : self.cur_val_loss
        }

        torch.save(checkpoint, path)
        self.logger.info("Saved model to checkpoint: " + path)

        with open(os.path.join(self.checkpoint_dir, 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.config, f=f)

    def _get_batch(self, batch):
        features = batch['input_features'].to(self.device)
        labels = batch['seq_class_ids'].to(self.device)
        n_time_steps = batch['n_time_steps'].to(self.device)
        phone_seq_lens = batch['phone_seq_lens'].to(self.device)
        day_indicies = batch['day_indicies'].to(self.device)

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
        
        patch_size = self.config['dataset']

        trains_losses = []
        val_losses = []

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
                self.logger.info(f"Running val after {i+1} batches")
                start_time = time.time()
                val_metrics = self.validation(
                    val_loader=val_loader
                )

    def validation(self, val_loader):
        self.model.eval()





def training_pipeline(config, logger):
    train_conf = config['train']
    pin_memory = train_conf['pin_memory']
    shuffle = train_conf['shuffle_data']
    n_data_workers = train_conf['n_data_workers']

    train_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='val', logger=logger)

    
    

    
    train_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='train', logger=logger)
    train_trials, _ = train_test_split_indices(
        config=config,
        file_paths=train_filepaths,
        logger=logger
    )
    train_ds = BrainToTextDataset(
        trial_indices=train_trials,
        config=config,
        logger=logger,
        kind='train'
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=None,
        shuffle=shuffle,
        num_workers=n_data_workers,
        pin_memory=pin_memory
    )

    val_filepaths = get_all_files(config['dataset']['info']['dataset_dir'], kind='val', logger=logger)
    val_trails, _ = train_test_split_indices(
        config=config,
        file_paths=val_filepaths,
        logger=logger
    )
    val_ds = BrainToTextDataset(
        trial_indices=val_trails,
        config=config,
        logger=logger,
        kind='test'
    )
    val_loader =  DataLoader(
        val_ds, 
        batch_size=None,
        shuffle=False,
        num_workers=n_data_workers,
        pin_memory=pin_memory
    )
    

    model = GRUDecoder(
        neural_dim=512,
        n_units=768,
        n_days=45,
        n_classes=41,
        rnn_dropout=0.1,
        input_dropout=0.1,
        n_layers=5,
        patch_size=14,
        patch_stride=4
    )
    
    trainer = Trainer(
        config=config,
        logger=logger,
        model=model
    )

