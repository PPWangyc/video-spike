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

        self.model_class = self.config.model.model_class
        self.metric = 'r2'        
        self.session_active_neurons = {}    

    
    def _forward_model_outputs(self, batch):
                
        batch = move_batch_to_device(batch, self.accelerator.device)

        return self.model(batch['video'])

    
    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_avg_spike_r2 = -torch.tensor(float('inf'))
        print("start training")
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:

                if eval_epoch_results[f'eval_avg_spike_r2'] > best_eval_avg_spike_r2:
                    best_eval_avg_spike_r2 = eval_epoch_results['eval_avg_spike_r2']
                    print(f"epoch: {epoch} best trial avg spike r2: {best_eval_avg_spike_r2}")
                    self.save_model(name="best_spike", epoch=epoch)
                    wandb.log({"best_spike_epoch": epoch})
                    
                    # https://discuss.pytorch.org/t/saving-model-and-optimiser-and-scheduler/52030/8
                    # if len(self.eid_list) > 1:
                    #     ckpt = { 
                    #         'epoch': epoch,
                    #         'model': self.model,
                    #         'optimizer': self.optimizer,
                    #         'lr_sched': self.lr_scheduler}
                    #     torch.save(ckpt, 'ckpt.pth')

                    for mod in self.modal_filter['output']:
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0][mod], 
                            preds=eval_epoch_results['eval_preds'][0][mod], 
                            epoch=epoch,
                            active_neurons=next(iter(self.session_active_neurons.values()))[mod][:5],
                            modality=mod
                        )
                        if self.config.wandb.use:
                            wandb.log(
                                {"best_epoch": epoch,
                                 f"best_gt_pred_fig_{mod}": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                 f"best_r2_fig_{mod}": wandb.Image(gt_pred_fig['plot_r2'])}
                            )
                        else:
                            gt_pred_fig['plot_gt_pred'].savefig(
                                os.path.join(self.log_dir, f"best_gt_pred_fig_{mod}_{epoch}.png")
                            )
                            gt_pred_fig['plot_r2'].savefig(
                                os.path.join(self.log_dir, f"best_r2_fig_{mod}_{epoch}.png")
                            )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} trial avg {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")
                if self.config.model.use_contrastive:
                    print(f"epoch: {epoch} eval s2b acc: {eval_epoch_results['eval_s2b_acc']} eval b2s acc: {eval_epoch_results['eval_b2s_acc']}")

            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                for mod in self.modal_filter['output']:
                    # take the first session for plotting
                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0][mod], 
                        preds=eval_epoch_results['eval_preds'][0][mod], 
                        epoch=epoch, 
                        modality=mod,
                        active_neurons=next(iter(self.session_active_neurons.values()))[mod][:5]
                    )
                    if self.config.wandb.use:
                        wandb.log({
                            f"gt_pred_fig_{mod}": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            f"r2_fig_{mod}": wandb.Image(gt_pred_fig['plot_r2'])
                        })
                    else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"gt_pred_fig_{mod}_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"r2_fig_{mod}_{epoch}.png")
                        )

            if self.config.wandb.use:
                wandb.log({
                    "train_loss": train_epoch_results['train_loss'],
                    "train_spike_loss": train_epoch_results['train_spike_loss'],
                    "train_behave_loss": train_epoch_results['train_behave_loss'],
                    "eval_loss": eval_epoch_results['eval_loss'],
                    "eval_spike_loss": eval_epoch_results['eval_spike_loss'],
                    "eval_behave_loss": eval_epoch_results['eval_behave_loss'],
                    #####
                    "train_static_loss": train_epoch_results['train_static_loss'],
                    "eval_static_loss": eval_epoch_results['eval_static_loss'],
                    #####
                    f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}'],
                    "eval_avg_spike_r2": eval_epoch_results[f'eval_avg_spike_r2'],
                    "eval_avg_behave_r2": eval_epoch_results[f'eval_avg_behave_r2'],
                    #####
                    "eval_avg_static_acc": eval_epoch_results[f'eval_avg_static_acc'],
                    "eval_avg_choice_acc": eval_epoch_results[f'eval_avg_choice_acc'],
                    "eval_avg_block_acc": eval_epoch_results[f'eval_avg_block_acc'],
                    #####
                    f"eval_s2b_acc": eval_epoch_results['eval_s2b_acc'],
                    f"eval_b2s_acc": eval_epoch_results['eval_b2s_acc'],
                })
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            #####
            wandb.log({"best_eval_loss": best_eval_loss,
                       "best_eval_avg_spike_r2": best_eval_avg_spike_r2,
                      }
                     )
            #####

    def computer_loss(self, outputs, batch):
        loss = self.criterion(outputs, batch['ap']).mean()
        return loss
    def train_epoch(self, epoch):
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
            steps += 1

        return{
            "train_loss": np.mean(train_loss),
            "lr": self.optimizer.param_groups[0]['lr']
        }
    
    
    def eval_epoch(self):
        
        self.model.eval()
        eval_loss = 0.
        eval_spike_loss = 0.
        eval_behave_loss = 0.
        #####
        eval_static_loss = 0.
        #####
        session_results = {}
        for eid in self.eid_list:
            session_results[eid] = {}
            for mod in self.modal_filter['output']:
                #####
                if mod == 'ap':
                    session_results[eid][mod] = {"gt": [], "preds": []}
                elif mod == 'behavior':
                    session_results[eid][mod] = {
                        "gt": [], "preds": [], 
                        "gt_choice": [], "preds_choice": [], 
                        "gt_block": [], "preds_block": [], 
                    }
                #####
            session_results[eid]['s2b_acc'] = []
            session_results[eid]['b2s_acc'] = []

        if self.eval_dataloader:

            with torch.no_grad():  

                if 'ap' in self.modal_filter['output']:
                    
                    for batch in self.eval_dataloader:
                        
                        if self.config.training.mask_type == "input":
                            self.masking_mode = random.sample(self.masking_schemes, 1)[0]
                        
                        outputs = self._forward_model_outputs(
                            batch, masking_mode=self.masking_mode, training_mode="encoding"
                        )
                        loss = outputs.loss
                        eval_loss += loss.item()
                        eval_spike_loss += outputs.mod_loss['ap'].item()
                        num_neuron = batch['spikes_data'].shape[2] 
                        eid = batch['eid'][0]

                        session_results[eid]['ap']["gt"].append(
                            outputs.mod_targets['ap'].clone()[:,:,:num_neuron]
                        )
                        session_results[eid]['ap']["preds"].append(
                            outputs.mod_preds['ap'].clone()[:,:,:num_neuron]
                        )
    
                        if outputs.contrastive_dict:
                            session_results[eid]['b2s_acc'].append(outputs.contrastive_dict['b2s_acc'])
                            session_results[eid]['s2b_acc'].append(outputs.contrastive_dict['s2b_acc'])

                if 'behavior' in self.modal_filter['output']:
                    
                    for batch in self.eval_dataloader:
                        
                        if self.config.training.mask_type == "input":
                            self.masking_mode = random.sample(self.masking_schemes, 1)[0]
                        
                        outputs = self._forward_model_outputs(
                            batch, masking_mode=self.masking_mode, training_mode="decoding"
                        )
                        loss = outputs.loss
                        eval_loss += loss.item()
                        #####
                        eval_behave_loss += outputs.mod_loss['dynamic'].item()
                        eval_static_loss += outputs.mod_loss['static'].item()
                        #####
                        num_neuron = batch['spikes_data'].shape[2] 
                        eid = batch['eid'][0]

                        session_results[eid]['behavior']["gt"].append(
                            outputs.mod_targets['behavior'].clone()
                        )
                        session_results[eid]['behavior']["preds"].append(
                            outputs.mod_preds['behavior'].clone()
                        )
                        #####
                        session_results[eid]['behavior']["gt_choice"].append(
                            outputs.targets_static['choice'].clone()
                        )
                        session_results[eid]['behavior']["preds_choice"].append(
                            outputs.preds_static['choice'].clone()
                        )
                        session_results[eid]['behavior']["gt_block"].append(
                            outputs.targets_static['block'].clone()
                        )
                        session_results[eid]['behavior']["preds_block"].append(
                            outputs.preds_static['block'].clone()
                        )
                        #####
    
                        if outputs.contrastive_dict:
                            session_results[eid]['b2s_acc'].append(outputs.contrastive_dict['b2s_acc'])
                            session_results[eid]['s2b_acc'].append(outputs.contrastive_dict['s2b_acc'])

            
            gt, preds, s2b_acc_list, b2s_acc_list = {}, {}, [], []
            spike_r2_results_list, behave_r2_results_list = [], []
            #####
            gt_static, preds_static = {}, {}
            choice_acc_results_list, block_acc_results_list = [], []
            #####
            for idx, eid in enumerate(self.eid_list):
                gt[idx], preds[idx] = {}, {}
                gt_static[idx], preds_static[idx] = {}, {}
                for mod in self.modal_filter['output']:
                    #####
                    if mod == 'behavior':
                        _gt = torch.cat(session_results[eid][mod]["gt"], dim=0)[:,:,:2]
                    else:
                        _gt = torch.cat(session_results[eid][mod]["gt"], dim=0)
                    #####
                    _preds = torch.cat(session_results[eid][mod]["preds"], dim=0)
                    if mod == 'ap' and 'ap' in self.modal_filter['output']:
                        _preds = torch.exp(_preds)
                    gt[idx][mod] = _gt
                    preds[idx][mod] = _preds
                    #####
                    if mod == 'behavior':
                        _gt_choice = torch.cat(session_results[eid][mod]["gt_choice"], dim=0)
                        _preds_choice = torch.cat(session_results[eid][mod]["preds_choice"], dim=0)
                        _gt_block = torch.cat(session_results[eid][mod]["gt_block"], dim=0)
                        _preds_block = torch.cat(session_results[eid][mod]["preds_block"], dim=0)
                        gt_static[idx]['choice'] = _gt_choice
                        preds_static[idx]['choice'] = _preds_choice
                        gt_static[idx]['block'] = _gt_block
                        preds_static[idx]['block'] = _preds_block
                    #####
                    
                if eid not in self.session_active_neurons:
                    self.session_active_neurons[eid] = {}

                for mod in self.modal_filter['output']:
                    
                    if mod == 'ap':
                        active_neurons = np.arange(gt[idx][mod].shape[-1]).tolist()
                        self.session_active_neurons[eid][mod] = active_neurons
                        
                    if mod == 'behavior':
                        self.session_active_neurons[eid]['behavior'] = [i for i in range(gt[idx]['behavior'].size(2))]
                    
                    if mod == 'ap':
                        results = metrics_list(
                            gt = gt[idx][mod][:,:,self.session_active_neurons[eid][mod]].transpose(-1,0),
                            pred = preds[idx][mod][:,:,self.session_active_neurons[eid][mod]].transpose(-1,0), 
                            metrics=["bps"], 
                            device=self.accelerator.device
                        )
                        spike_r2_results_list.append(results["bps"])
                      
                    elif mod == 'behavior':
                        results = metrics_list(gt = gt[idx][mod],
                                            pred = preds[idx][mod],
                                            metrics=["rsquared"],
                                            device=self.accelerator.device)
                        behave_r2_results_list.append(results["rsquared"])

                        #####
                        from sklearn.metrics import balanced_accuracy_score
                        choice_acc_results_list.append(balanced_accuracy_score(
                            gt_static[idx]['choice'].cpu().numpy(), preds_static[idx]['choice'].cpu().numpy()
                        ))
                        block_acc_results_list.append(balanced_accuracy_score(
                            gt_static[idx]['block'].cpu().numpy(), preds_static[idx]['block'].cpu().numpy()
                        ))
                        #####
                    
                if self.config.model.use_contrastive:
                    assert len(session_results[eid]['s2b_acc']) == len(session_results[eid]['b2s_acc'])
                    assert len(session_results[eid]['s2b_acc']) > 0
                    s2b_acc_list.append(np.mean(session_results[eid]['s2b_acc']))
                    b2s_acc_list.append(np.mean(session_results[eid]['b2s_acc']))
                else:
                    s2b_acc_list = [0]
                    b2s_acc_list = [0]

        spike_r2 = np.nanmean(spike_r2_results_list)
        behave_r2 = np.nanmean(behave_r2_results_list)
        choice_acc = np.nanmean(choice_acc_results_list)
        block_acc = np.nanmean(block_acc_results_list)

        return {
            "eval_loss": eval_loss/len(self.eval_dataloader),
            "eval_spike_loss": eval_spike_loss/len(self.eval_dataloader),
            "eval_behave_loss": eval_behave_loss/len(self.eval_dataloader),
            #####
            "eval_static_loss": eval_static_loss/len(self.eval_dataloader),
            f"eval_trial_avg_{self.metric}": np.nanmean([spike_r2, behave_r2, choice_acc, block_acc]),
             #####
            "eval_avg_spike_r2": spike_r2,
            "eval_avg_behave_r2": behave_r2,
            #####
            "eval_avg_static_acc": np.nanmean([choice_acc, block_acc]),
            "eval_avg_choice_acc": choice_acc,
            "eval_avg_block_acc": block_acc,
            #####
            "eval_gt": gt,
            "eval_preds": preds,
            "eval_s2b_acc": np.mean(s2b_acc_list),
            "eval_b2s_acc": np.mean(b2s_acc_list)
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