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
                wandb.log(step_logs) if self.use_wandb else self.log.info('{}'.format(step_logs))
                loss = step_logs['loss']
                current_step += 1
                if best_loss > loss:
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
        outputs = self._inferene(batch)
        ref = outputs['ref']
        pos = outputs['pos']
        neg = outputs['neg']
        loss_dict = self.criterion(ref, pos, neg)
        self.accelerator.backward(loss_dict['loss'])
        self.optimizer.step()
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        return {
            'cur_step': cur_step,
            **loss_dict
        }
    
    def _inferene(self, batch):
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
        # use best model to transform the data
        self.log.info('Transforming the data')
        self.model.load_state_dict(self.best_model)
        data_loader, self.model = self.accelerator.prepare(data_loader, self.model)
        features = []
        for batch in tqdm(data_loader):
            outputs = self._forward(batch['ref'])
            if 'z' in outputs:
                embedding = outputs['z']
            else:
                self.log.error('No embedding found in the model!')
            features.append(embedding)
        return torch.cat(features, dim=0)
    
    def _prepare_accelerator(self):
        if self.accelerator is not None:
            self.log.info('Preparing accelerator!')
            self.data_loader, self.model, self.optimizer = self.accelerator.prepare(
                self.data_loader, self.model, self.optimizer
            )
        else:
            self.log.warning('No accelerator provided, using CPU!')

    def _initit_log(self, kwargs):
        self.log = kwargs.get('log', None)
        eid = kwargs.get('eid', None)
        model_name = self.model.__class__.__name__
        self.use_wandb = kwargs.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project='video-ssl', 
                       name="{}_{}".format(eid[:5], model_name),
            )
        else:
            self.log.warning('Not using wandb!')
            self.log.info(f'Experiment ID: {eid}, Model: {model_name}, Max steps: {self.max_steps}')