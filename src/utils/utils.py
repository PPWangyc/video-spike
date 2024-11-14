import argparse
import os
import torch
import numpy as np
import random
from model.linear import Linear
from model.videomae import VideoMAE
import matplotlib.pyplot as plt
from utils.metric_utils import r2_score, bits_per_spike
from sklearn.metrics import r2_score as r2_score_sklearn
from sklearn.metrics import accuracy_score
import glob
import pandas as pd

NAME2MODEL = {
    "Linear": Linear,
    "VideoMAE": VideoMAE
}

def get_args():
    parser = argparse.ArgumentParser(description='IBL Spike Video Project')
    parser.add_argument('--model_config', type=str, default='configs/model/model_config.yaml', help='Model config file')
    parser.add_argument('--train_config', type=str, default='configs/train/train_config.yaml', help='Train config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--eid', type=str, default='d57df551-6dcb-4242-9c72-b806cff5613a')

    args = parser.parse_args()
    return args

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def plot_gt_pred(gt, pred, epoch=0,modality="behavior"):
    # plot Ground Truth and Prediction in the same figur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}, Mod: {}".format(epoch, 
                                             modality))
    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[],modality="behavior"):
    # Create one figure and axis for all plots
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for neuron in neuron_idx:
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[neuron_idx.index(neuron)]
        ax.plot(gt[:, neuron].cpu().numpy(), label="Ground Truth", color="blue")
        ax.plot(pred[:, neuron].cpu().numpy(), label="Prediction", color="red")
        ax.set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax.legend()
        # x label
        ax.set_xlabel("Time")
        # y label
        ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Mod: {}, Avg R2: {:.4f}".format(epoch, 
                                                            modality, 
                                                            np.mean(r2_values)))
    return fig

# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}

    if "bps" in metrics:
        _gt, _pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        bps_list = []
        for i in range(gt.shape[-1]): 
            bps = bits_per_spike(_pred[:,:,[i]], _gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    
    if "r2" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2s = [r2_score(y_true=gt[i].T[k], y_pred=pred[i].T[k], device=device) for k in range(len(gt[i].T))]
            r2_list.append(np.ma.masked_invalid(r2s).mean())
        r2 = np.mean(r2_list)
        results["r2"] = r2
        
    if "behave_r2" in metrics:
        r2_list = []
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        for i in range(gt.shape[0]):
           r2 = r2_score_sklearn(gt[i].flatten(), pred[i].flatten())
           r2_list.append(r2)
        mean_r2 = np.nanmean(r2_list)
        results["behave_r2"] = mean_r2
        
    if "rsquared" in metrics:
        r2 = 0
        _gt, _pred = gt.cpu().clone(), pred.cpu().numpy()
        r2_list = []
        for i in range(gt.shape[-1]):
            r2 = r2_score_sklearn(y_true=_gt[:,:,i], y_pred=_pred[:,:,i])
            r2_list.append(r2)
        # for i in range(gt.shape[-1]):
        #     r2_list = []
        #     for j in range(gt.shape[0]):
        #         r2 = r2_score_sklearn(y_true=_gt[j,:,i], y_pred=_pred[j,:,i])
        #         r2_list.append(r2)
        #     r2 += np.nanmean(r2_list)
        # results["rsquared"] = r2 / gt.shape[-1]
        results["rsquared"] = np.nanmean(r2_list)
        
    if "mse" in metrics:
        mse = torch.mean((gt - pred) ** 2)
        results["mse"] = mse
        
    if "mae" in metrics:
        mae = torch.mean(torch.abs(gt - pred))
        results["mae"] = mae
        
    if "acc" in metrics:
        acc = accuracy_score(gt.cpu().numpy(), pred.cpu().detach().numpy())
        results["acc"] = acc

    return results

def get_log(log_dir):
    # recursively get all log files in the log directory
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".npy"):
                log_files.append(os.path.join(root, file))
    df_log = {}
    for log_file in log_files:
        data = np.load(log_file, allow_pickle=True).item()
        eid = log_file.split("results")[1].split("/")[1]
        df_log[log_file] = data['test_res']
        df_log[log_file]["eid"] = eid
        mod = log_file.split("results")[1].split("/")[2]
        df_log[log_file]["mod"] = mod
    df_log = pd.DataFrame(df_log)
    df_log = df_log.T
    return df_log

def draw_results(df_log, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"]):
    # draw results
    # only take columns test_bps, eid, mod
    df_log = df_log[["test_" + metric for metric in metrics] + ["eid", "mod"]]
    # remove eid c7bf2
    df_log = df_log[df_log["eid"] != "03d9a"]
    # group by mod
    df_grouped = df_log.groupby("mod")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for mod, group in df_grouped:
        # create a box plot for each modality bps
        bps = group["test_bps"].values
        # make a box plot of the bps of each modality, make the color of the median black
        ax.boxplot(bps, positions=[list(df_grouped.groups.keys()).index(mod)], widths=0.2, medianprops=dict(color="black"))
        # make the bar plot of the mean of the bps of each modality
        ax.bar(list(df_grouped.groups.keys()).index(mod), np.nanmean(bps), width=0.3)
        # plot mean value on the top of the bar
        ax.text(list(df_grouped.groups.keys()).index(mod), np.nanmean(bps), "{:.2f}".format(np.nanmean(bps)), ha="center", va="bottom")
    # label x axis
    ax.set_xticklabels(df_grouped.groups.keys())
    # label y axis
    ax.set_ylabel("bps")
    return fig
