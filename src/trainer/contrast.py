import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
from transformers import AutoImageProcessor
import time
def _get_input_modailities(config):
    input_modalities = []
    avail_mod = config.data.modalities.keys()
    for mod in avail_mod:
        if config.data.modalities[mod]['input']:
            input_modalities.append(mod)
    return input_modalities
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

        self.accelerator = kwargs.get('accelerator', None)
        self.criterion = kwargs.get('criterion', None)
        self.max_steps = kwargs.get('max_steps', 1000)
        self.log = kwargs.get('log', None)
        self._unfreeze()

    def fit(self):
        self.log.info('Starting fitting!')
        current_step = 0
        self.model.train()
        best_loss = np.inf
        start = time.time()
        while current_step < self.max_steps:
            for batch in self.data_loader:
                loss = self.step(batch)
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

    def step(self, batch):
        self.model.train()
        batch = move_batch_to_device(batch, self.accelerator.device)
        self.optimizer.zero_grad()
        outputs = self._inferene(batch)
        ref = outputs['ref']
        pos = outputs['pos']
        neg = outputs['neg']
        loss = self.criterion(ref, pos, neg)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
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
        features = []
        for batch in data_loader:
            batch = move_batch_to_device(batch, self.accelerator.device)
            embedding = self._forward(batch['ref'])
            features.append(embedding)
            print(embedding.shape)
        return torch.cat(features, dim=0)