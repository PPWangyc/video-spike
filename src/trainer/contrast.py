import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, train_rrr
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
        self.val_data_loader = kwargs.get('val_data_loader', None)
        self.train_data_loader = kwargs.get('train_data_loader', None)
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
        best_bps = -np.inf
        start = time.time()
        while current_step < self.max_steps:
            for batch in self.data_loader:
                step_logs = self._step(batch, current_step)
                if self.accelerator.is_main_process:
                    wandb.log(step_logs) if self.use_wandb else self.log.info('{}'.format(step_logs))
                current_step += 1
                if current_step >= self.max_steps:
                    break
            val_res = self._validate()
            if self.accelerator.is_main_process:
                wandb.log(val_res) if self.use_wandb else self.log.info('{}'.format(val_res))
                if val_res['val_bps'] > best_bps:
                    best_bps = val_res['val_bps']
                    wandb.log({'best_val_bps': best_bps}) if self.use_wandb else self.log.info(f'Best val bps: {best_bps}')
                    self._save_model(os.path.join(self.log_dir, 'best_model.pth'))
        end = time.time()
        self.log.info(f'Training took: {end-start} seconds')
        return best_loss
    
    @torch.no_grad()
    def _save_model(self, path):
        self.log.info(f'Saving the model to: {path}')
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            self.log.error(f'Error saving the model: {e}')

    @torch.no_grad()
    def _load_model(self, path):
        self.log.info(f'Loading the model from: {path}')
        # check if the path exists
        if not os.path.exists(path):
            self.log.warning(f'Path does not exist: {path}')
            return None
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path))
        return True

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

    def _validate(self):
        self.log.info('Validating the model')
        with torch.no_grad():
            self.model.eval()
            train_embeddings, train_y = self.transform(self.train_data_loader, return_neural=True, use_best=True)
            val_embeddings, val_y = self.transform(self.val_data_loader, return_neural=True, use_best=True)
            train_n, val_n = train_y.shape[0], val_y.shape[0]
            e_dim = train_embeddings.shape[-1]
            # reshape the embeddings
            train_embeddings = train_embeddings.view(train_n, -1, e_dim)
            val_embeddings = val_embeddings.view(val_n, -1, e_dim)
            # randomly select 100 idx from 0 to 119 sorted
            idx = np.random.choice(119, 100, replace=False)
            sorted_idx = np.sort(idx)
            train_embeddings = train_embeddings[:, sorted_idx, :].cpu().numpy()
            val_embeddings = val_embeddings[:, sorted_idx, :].cpu().numpy()
            train_y = train_y.cpu().numpy()
            val_y = val_y.cpu().numpy()
        # form data_dict for rrr encoding model
        data_dict = {
            self.eid: {
                'X': [train_embeddings, val_embeddings],
                'y': [train_y, val_y],
                'setup': {}
            }
        }
        # make mask ratio 0
        rrr_result = train_rrr(data_dict)
        val_bps = np.nanmean(rrr_result[self.eid]['bps'])
        self.log.info(f'Validation bps: {val_bps}')
        val_dict = {
            'val_bps': val_bps
        }
        return val_dict

    @torch.no_grad()
    def transform(self, 
                  data_loader,
                  use_best=False,
                  return_neural=False):
        neurals = []
        if use_best:
            self.log.info('Loading the best model for transformation')
            if not self._load_model(os.path.join(self.log_dir, 'best_model.pth')):
                self.log.warning('Model not loaded, using the last model!')
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
                neurals.append(batch['neural'])
                if len(batch['ref'].shape) == 5:
                    batch['ref'] = batch['ref'].squeeze(0)
                outputs = self._forward(batch['ref'])
                if 'z' in outputs:
                    embedding = outputs['z']
                else:
                    self.log.error('No embedding found in the model!')
                features.append(embedding)
            features = torch.cat(features, dim=0)
            neurals = torch.cat(neurals, dim=0)
            # set the mask ratio back to original
            if self.distribute:
                self.model.module.config.mask_ratio = self.mask_ratio
            else:
                self.model.config.mask_ratio = self.mask_ratio
            # features = self.accelerator.gather(features) # gather all the features from all the processes
            if return_neural:
                return features, neurals
            return features
    
    def _prepare_accelerator(self):
        if self.accelerator is not None:
            self.log.info('Preparing accelerator!')
            self.data_loader, self.model, self.optimizer, self.criterion = self.accelerator.prepare(
                self.data_loader, self.model, self.optimizer, self.criterion
            )
            self.val_data_loader = self.accelerator.prepare(self.val_data_loader) if self.val_data_loader is not None else None
            self.train_data_loader = self.accelerator.prepare(self.train_data_loader) if self.train_data_loader is not None else None
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
        self.log_dir = os.path.join('logs', eid, model_name, str(self.max_steps))
        os.makedirs(self.log_dir, exist_ok=True)
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(project='video-ssl', 
                       name="{}_{}".format(eid[:5], model_name),
            )
        else:
            self.log.warning('Not using wandb!')
            self.log.info(f'Experiment ID: {eid}, Model: {model_name}, Max steps: {self.max_steps}')
        self.eid = eid
        self._set_model_mask_ratio()

    def _set_model_mask_ratio(self):
        if self.distribute:
            self.mask_ratio = self.model.module.config.mask_ratio
        else:
            self.mask_ratio = self.model.config.mask_ratio