import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import time
import wandb

class ContrastTrainer():
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        **kwargs
    ):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.accelerator = kwargs.get('accelerator', None)
        self.max_steps = kwargs.get('max_steps', 1000)
        self.criterion = kwargs.get('criterion', None)

        # init logger and wandb
        self._initit_log(kwargs)
        # unfreeze the model params
        self._unfreeze()
        # prepare accelerator
        self._prepare_accelerator()
        
    def fit(self):
        self.log.info('Starting fitting!')
        current_step = 0
        self.model.train()
        best_loss = np.inf
        start = time.time()
        while current_step < self.max_steps:
            for batch in self.data_loader:
                step_logs = self._step(batch, current_step)
                if self.accelerator.is_main_process:
                    wandb.log(step_logs) if self.use_wandb else self.log.info('{}'.format(step_logs))
                loss = step_logs['loss']
                current_step += 1
                if best_loss > loss and self.accelerator.is_main_process:
                    best_loss = loss
                    self.log.info(f'Best loss: {best_loss} at step: {current_step}')
                    self.best_model = self.model.state_dict()
                if current_step >= self.max_steps:
                    break
        end = time.time()
        self.log.info(f'Training took: {end-start} seconds')
        return best_loss

    def _step(self, batch, cur_step):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self._inference(batch)
        ref = outputs['ref']
        pos = outputs['pos']
        neg = outputs['neg']
        loss_dict = self.criterion(ref, pos, neg)
        self.accelerator.backward(loss_dict['loss'])
        self.optimizer.step()
        # self.lr_scheduler.step()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        return {
            'cur_step': cur_step, # current step
            **loss_dict, # loss, loss_pos, loss_neg
            'lr': self.optimizer.param_groups[0]['lr'], # learning rate
            # 'temperature': self.criterion.info_nce.temperature # temperature for infoNCE loss
            'temperature': ref['temp'] if 'temp' in ref else 0
        }
    
    def _inference(self, batch):
        ref = batch['ref']
        pos = batch['pos']
        neg = batch['neg']
        ref = self._forward(ref)
        pos = self._forward(pos)
        neg = self._forward(neg)
        return {
            'ref': ref,
            'pos': pos,
            'neg': neg
        }
        
    def _forward(self, image):
        outputs = self.model(image)
        return outputs
    
    def _unfreeze(self):
        self.log.info('Unfreezing the model')
        for param in self.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def transform(self, data_loader):
        if self.accelerator.is_main_process:
            self.log.info('Transforming the data')
            if self.distribute:
                self.log.warning(f'Using DDP, setting mask ratio to 0, original mask ratio: {self.model.module.config.mask_ratio}')
                self.model.module.config.mask_ratio = 0
                self.log.warning(f'Moving model to device: {self.accelerator.device}')
                self.model = self.model.to(self.accelerator.device)
            else:
                self.model.config.mask_ratio = 0
            self.model.eval()
            features = []
            for batch in tqdm(data_loader):
                batch = move_batch_to_device(batch, self.accelerator.device)
                outputs = self._forward(batch['ref'])
                if 'z' in outputs:
                    embedding = outputs['z']
                else:
                    self.log.error('No embedding found in the model!')
                features.append(embedding)
            features = torch.cat(features, dim=0)
            # features = self.accelerator.gather(features) # gather all the features from all the processes
            return features
    
    def _prepare_accelerator(self):
        if self.accelerator is not None:
            self.log.info('Preparing accelerator!')
            self.data_loader, self.model, self.optimizer, self.criterion = self.accelerator.prepare(
                self.data_loader, self.model, self.optimizer, self.criterion
            )
            # Print memory usage in bytes
            self.log.info(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            self.log.info(f"Max Memory Allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
            self.log.info(f"Memory Cached (Reserved): {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        else:
            self.log.warning('No accelerator provided, using CPU!')

    def _initit_log(self, kwargs):
        self.log = kwargs.get('log', None)
        eid = kwargs.get('eid', None)
        model_name = self.model.__class__.__name__
        self.use_wandb = kwargs.get('use_wandb', False)
        self.distribute = not self.accelerator.state.distributed_type.value == 'NO'
        self.log.warning('Using DDP!') if self.distribute else self.log.warning('Single GPU training!')
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(project='video-ssl', 
                       name="{}_{}".format(eid[:5], model_name),
            )
        else:
            self.log.warning('Not using wandb!')
            self.log.info(f'Experiment ID: {eid}, Model: {model_name}, Max steps: {self.max_steps}')