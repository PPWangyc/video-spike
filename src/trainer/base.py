import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random

class BaseTrainer():
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = kwargs.get("criterion", None)

        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.dataset_split_dict = kwargs.get("dataset_split_dict", None)

        self.model_class = self.config.model.model_class
        self.metric = ['bps']       
        self.session_active_neurons = {}

        self._create_log_dir()

    def _create_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)
        wandb.init(project=self.config.wandb.project, 
                   config=self.config) if self.config.wandb.use else None

    
    def _forward_model_outputs(self, batch):
                
        batch = move_batch_to_device(batch, self.accelerator.device)
        return self.model(batch['video'])

    def _plot_figs(self, eval_epoch_results, epoch):
        gt_pred_fig = self.plot_epoch(
            gt=eval_epoch_results['eval_gt'][0], 
            preds=eval_epoch_results['eval_preds'][0], 
            epoch=epoch,
            active_neurons=range(5),
            modality='ap'
        )
        if self.config.wandb.use:
            wandb.log(
                {
                    "best_epoch": epoch,
                    f"best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                    f"best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                }
            )
        else:
            gt_pred_fig['plot_gt_pred'].savefig(
                os.path.join(self.log_dir, f"best_trial_{epoch}.png")
            )
            gt_pred_fig['plot_r2'].savefig(
                os.path.join(self.log_dir, f"best_neuron_{epoch}.png")
            )
    
    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_bps = -torch.tensor(float('inf'))
        print("start training")
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch()
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results['eval_res']['eval_bps'] > best_eval_bps:
                    best_eval_bps = eval_epoch_results['eval_res']['eval_bps']
                    print(f"epoch: {epoch} best eval_bps: {best_eval_bps}")
                    self.save_model(name="best", epoch=epoch)

                    wandb.log({"best_eval_bps_epoch": epoch}) if self.config.wandb.use else None
                    self._plot_figs(eval_epoch_results, epoch)
                    print(f"best_epoch: {epoch}, best_eval_bps: {best_eval_bps}")

            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                self._plot_figs(eval_epoch_results, epoch)

            if self.config.wandb.use:
                wandb.log({
                    **train_epoch_results,
                    **eval_epoch_results['eval_res'],
                })
            else:
                print({**train_epoch_results, **eval_epoch_results['eval_res']})
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            #####
            wandb.log({"best_eval_loss": best_eval_loss,
                       "best_eval_bps": best_eval_bps,
                      }
                     )
            #####

    def computer_loss(self, outputs, batch):
        loss = self.criterion(outputs, batch['ap']).mean()
        return loss
    def train_epoch(self):
        train_loss = []
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            outputs = self._forward_model_outputs(batch)
            loss = self.computer_loss(outputs, batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss.append(loss.item())

        return{
            "train_loss": round(np.mean(train_loss),5),
            "lr": self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        eval_loss = []
        session_results = {}
        for eid in self.dataset_split_dict['eid']['val']:
            session_results[eid] = {'gt': [], 'preds': []}
        if self.eval_dataloader is not None:
            with torch.no_grad():                      
                for batch in self.eval_dataloader:
                    outputs = self._forward_model_outputs(batch)
                    loss = self.computer_loss(outputs, batch)
                    # outputs = torch.exp(outputs)
                    eval_loss.append(loss.item())
                    eid = batch['eid'][0]

                    session_results[eid]["gt"].append(batch['ap'])
                    session_results[eid]["preds"].append(outputs)

            gt, preds = {}, {}
            metrics_results = {k: [] for k in self.metric}
            for idx, eid in enumerate(self.dataset_split_dict['eid']['val']):
                gt[idx], preds[idx] = {}, {}
                _gt = torch.cat(session_results[eid]["gt"], dim=0)
                _preds = torch.cat(session_results[eid]["preds"], dim=0)
                _preds = torch.exp(_preds)
                gt[idx] = _gt
                preds[idx] = _preds

                results = metrics_list(
                    gt = gt[idx].transpose(-1,0),
                    pred = preds[idx].transpose(-1,0), 
                    metrics=["bps"], 
                    device=self.accelerator.device
                )
                metrics_results['bps'].append(results['bps'])
        _metrics_results = {f"eval_{k}": round(np.mean(v),5) for k, v in metrics_results.items()}
        return {
            "eval_gt": gt,
            "eval_preds": preds,
            "eval_res":{
                "eval_loss": round(np.mean(eval_loss),5),
                **_metrics_results
                }
        }

    
    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        
        if modality == 'ap':
            gt_pred_fig = plot_gt_pred(
                gt = gt.mean(0).T.cpu().numpy(),
                pred = preds.mean(0).T.detach().cpu().numpy(),
                epoch = epoch,
                modality = modality
                )
        elif modality == 'behavior':
            gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                        pred = preds.mean(0).T.detach().cpu().numpy(),
                        epoch = epoch,
                        modality=modality)
            active_neurons = range(gt.size()[-1])
            
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=active_neurons,
                epoch = epoch)
        
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }

    
    def save_model(self, name="last", epoch=0):
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))