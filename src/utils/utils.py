import argparse
import os
import torch
import numpy as np
import random
from model.linear import Linear
from model.videomae import VideoMAE
from model.vit_mae.vit_mae import (
    ContrastViT,
    ContrastViTMAE,
    MAE
)
from model.rrr import (
    train_model,
    train_model_main
)
import matplotlib.pyplot as plt
from utils.metric_utils import r2_score, bits_per_spike
from sklearn.metrics import r2_score as r2_score_sklearn
from sklearn.metrics import accuracy_score
import glob
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cebra
from sklearn.decomposition import PCA

NAME2MODEL = {
    "Linear": Linear,
    "VideoMAE": VideoMAE,
    "ContrastViT": ContrastViT,
    "ContrastViTMAE": ContrastViTMAE,
    "MAE": MAE
}

def get_args():
    parser = argparse.ArgumentParser(description='IBL Spike Video Project')
    parser.add_argument('--model_config', type=str, default='configs/model/model_config.yaml', help='Model config file')
    parser.add_argument('--train_config', type=str, default='configs/train/train_config.yaml', help='Train config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--eid', type=str, default='d57df551-6dcb-4242-9c72-b806cff5613a')
    parser.add_argument('--input_mod', type=str, default='whisker-motion-energy', help='Input modality')
    parser.add_argument('--model', type=str, default='cm', help='Model name')
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
    torch.backends.cudnn.benchmark = False
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

def _std(arr):
    mean = np.mean(arr, axis=0) # (T, N)
    std = np.std(arr, axis=0) # (T, N)
    std = np.clip(std, 1e-8, None) # (T, N) 
    arr = (arr - mean) / std
    return arr, mean, std

def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:,:,i] = (arr == _uni)
    return ret

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

def get_rrr_data(dataloader, input_mod):
    X, y = [], []
    timestamps = []
    for batch in tqdm(dataloader):
        assert 'timestamp' in batch, "timestamp is not in the batch"
        timestamps.append(batch['timestamp'].numpy())
        if input_mod == 'whisker-of-video':
            x_vec = torch.tensor(np.median(batch[input_mod][...,0].numpy(),axis=(2,3)))
            y_vec = torch.tensor(np.median(batch[input_mod][...,1].numpy(),axis=(2,3)))
            # # normalize x and y to 0-1
            # for i in range(x_vec.shape[0]):
            #     x_vec[i] = (x_vec[i] - x_vec[i].min()) / (x_vec[i].max() - x_vec[i].min())
            #     y_vec[i] = (y_vec[i] - y_vec[i].min()) / (y_vec[i].max() - y_vec[i].min())
            # x_vec = batch[input_mod][...,0].mean(axis=(2,3))
            # y_vec = batch[input_mod][...,1].mean(axis=(2,3))
            value = torch.stack([x_vec, y_vec], dim=2)
            # took median of x and y
            
            X.append(value.numpy())
        elif input_mod == 'all':
            block = batch['block'].numpy()
            choice = batch['choice'].numpy()
            wheel_speed = batch['wheel-speed'].numpy()
            me = batch['whisker-motion-energy'].numpy()
            T = wheel_speed.shape[1]
            # repeat block and choice for T times
            block = np.repeat(block, T, axis=1)
            choice = np.repeat(choice, T, axis=1)
            me, wheel_speed, choice, block = np.expand_dims(me, axis=2), np.expand_dims(wheel_speed, axis=2), np.expand_dims(choice, axis=2), np.expand_dims(block, axis=2)
            value = np.concatenate([me, wheel_speed, choice, block], axis=2)
            X.append(value)
        elif input_mod == 'other':
            block = batch['block'].numpy()
            choice = batch['choice'].numpy()
            wheel_speed = batch['wheel-speed'].numpy()
            # for i in range(wheel_speed.shape[0]):
            #     wheel_speed[i] = (wheel_speed[i] - wheel_speed[i].min()) / (wheel_speed[i].max() - wheel_speed[i].min())
            T = wheel_speed.shape[1]
            # repeat block and choice for T times
            block = np.repeat(block, T, axis=1)
            choice = np.repeat(choice, T, axis=1)
            wheel_speed, choice, block = np.expand_dims(wheel_speed, axis=2), np.expand_dims(choice, axis=2), np.expand_dims(block, axis=2)
            value = np.concatenate([wheel_speed, choice, block], axis=2)
            X.append(value)
        elif input_mod == 'of-all':
            block = batch['block'].numpy()
            choice = batch['choice'].numpy()
            wheel_speed = batch['wheel-speed'].numpy()
            of_x = torch.tensor(np.median(batch['whisker-of-video'][...,0].numpy(),axis=(2,3)))
            of_y = torch.tensor(np.median(batch['whisker-of-video'][...,1].numpy(),axis=(2,3)))
            # for i in range(of_x.shape[0]):
            #     of_x[i] = (of_x[i] - of_x[i].min()) / (of_x[i].max() - of_x[i].min())
            #     of_y[i] = (of_y[i] - of_y[i].min()) / (of_y[i].max() - of_y[i].min())
            #     wheel_speed[i] = (wheel_speed[i] - wheel_speed[i].min()) / (wheel_speed[i].max() - wheel_speed[i].min())
            T = wheel_speed.shape[1]
            # repeat block and choice for T times
            block = np.repeat(block, T, axis=1)
            choice = np.repeat(choice, T, axis=1)
            of = torch.stack([of_x, of_y], dim=2)
            of_last = of[:,-1]
            of = torch.cat([of, of_last.unsqueeze(1)], dim=1)
            wheel_speed, choice, block = np.expand_dims(wheel_speed, axis=2), np.expand_dims(choice, axis=2), np.expand_dims(block, axis=2)
            value = np.concatenate([of, wheel_speed, choice, block], axis=2)
            X.append(value)
        elif input_mod == 'whisker-video':
            value = batch[input_mod].numpy()
            X.append(value)
        elif input_mod == 'wheel-speed':
            value = batch[input_mod].numpy()
            # for i in range(value.shape[0]):
            #     value[i] = (value[i] - value[i].min()) / (value[i].max() - value[i].min())
            X.append(value)
        else:
            X.append(batch[input_mod].numpy())
        y.append(batch["ap"].numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    timestamps = np.concatenate(timestamps, axis=0)
    return X, y, timestamps

def get_cebra_embedding(video, out_dim=3, save_path=None):
    # video: (N, T, C, H, W), C = 1 grayscale
    # output: N, T, out_dim
    video_data = video.squeeze(2)
    n, t, h, w = video_data.shape
    print(n,t,h,w)
    video_data = video_data.reshape(n*t, -1)
    single_cebra_model = cebra.CEBRA(batch_size=512,
                                    model_architecture='offset10-model',
                                    output_dimension=out_dim,
                                    max_iterations=5000,
                                    max_adapt_iterations=5000)
    single_cebra_model.fit(video_data)
    embedding = single_cebra_model.transform(video_data)
    assert(embedding.shape == (n*t, out_dim))

    if save_path:
        ax = cebra.plot_loss(single_cebra_model)
        fig = ax.get_figure()
        fig.savefig(save_path + "_loss.png")

        ax = cebra.plot_embedding(embedding)
        fig = ax.get_figure()
        fig.savefig(save_path + "_embedding.png")
    return embedding.reshape(n, t, out_dim)
    # cebra_embeddings = []
    # for i in tqdm(range(video.shape[0])):
    #     # (T, C, H, W) -> (T, D)
    #     video_data = video[i].squeeze(1)
    #     t, h, w = video_data.shape
    #     video_data = video_data.reshape(t, -1)
    #     # T, D
    #     single_cebra_model = cebra.CEBRA(batch_size=32,
    #                                     output_dimension=out_dim,
    #                                     max_iterations=5000,
    #                                     max_adapt_iterations=5000)
    #     single_cebra_model.fit(video_data)
    #     embedding = single_cebra_model.transform(video_data)
    #     assert(embedding.shape == (t, out_dim))
    #     cebra_embeddings.append(embedding)

    cebra_embeddings = np.array(cebra_embeddings)
    return cebra_embeddings

def get_pca_embedding(video, out_dim=5):
    # video: (N, T, C, H, W), C = 1 grayscale
    # output: N, T, out_dim
    video_data = video.squeeze(2)
    n, t, h, w = video_data.shape
    print(n,t,h,w)
    video_data = video_data.reshape(n*t, -1)
    pca = PCA(n_components=out_dim)
    pca_embeddings = pca.fit_transform(video_data)
    assert(pca_embeddings.shape == (n*t, out_dim))
    return pca_embeddings.reshape(n, t, out_dim)
    # pca_embeddings = []
    # for i in tqdm(range(video.shape[0])):
    #     # (T, C, H, W) -> (T, D)
    #     video_data = video[i].squeeze(1)
    #     t, h, w = video_data.shape
    #     video_data = video_data.reshape(t, -1)
    #     # T, D
    #     pca = PCA(n_components=out_dim)
    #     embedding = pca.fit_transform(video_data)
    #     assert(embedding.shape == (t, out_dim))
    #     pca_embeddings.append(embedding)

    # pca_embeddings = np.array(pca_embeddings)
    return pca_embeddings

def train_rrr(data_dict):
    ground_truth = {}
    for eid in data_dict:
        _, mean_X, std_X = _std(data_dict[eid]["X"][0])
        _, mean_y, std_y = _std(data_dict[eid]["y"][0])
        ground_truth[eid] = data_dict[eid]["y"][1].copy()
        for i in range(2):
            K = data_dict[eid]["X"][i].shape[0]
            T = data_dict[eid]["X"][i].shape[1]
            data_dict[eid]["X"][i] = (data_dict[eid]["X"][i] - mean_X) / std_X
            if len(data_dict[eid]["X"][i].shape) == 2:
                data_dict[eid]["X"][i] = np.expand_dims(data_dict[eid]["X"][i], axis=0)
            # add bias
            data_dict[eid]["X"][i] = np.concatenate([data_dict[eid]["X"][i], np.ones((K, T, 1))], axis=2)
            data_dict[eid]["y"][i] = (data_dict[eid]["y"][i] - mean_y) / std_y
            print(f"X shape: {data_dict[eid]['X'][i].shape}, y shape: {data_dict[eid]['y'][i].shape}")
        data_dict[eid]["setup"]["mean_X_Tv"] = mean_X
        data_dict[eid]["setup"]["std_X_Tv"] = std_X
        data_dict[eid]["setup"]["mean_y_TN"] = mean_y
        data_dict[eid]["setup"]["std_y_TN"] = std_y
    l2 = 100
    n_comp = 3
    print("Training RRR")
    result = {}
    test_bps = []
    for eid in data_dict:
        _train_data = {eid: data_dict[eid]}
        model, mse_val = train_model_main(
            train_data=_train_data,
            l2=l2,
            n_comp=n_comp,
            model_fname='tmp',
            save=False,
        )
        print(f"Model {eid} trained")
        with torch.no_grad():
            _, _, pred_orig = model.predict_y_fr(data_dict, eid, 1)
        pred = pred_orig.cpu().numpy()
        threshold = 1e-3
        trial_len = 2.

        pred = np.clip(pred, threshold, None)
        num_neuron = pred.shape[2]
        gt_held_out = ground_truth[eid]
        mean_fr = gt_held_out.sum(1).mean(0) / trial_len
        keep_idxs = np.arange(len(mean_fr)).flatten()

        bps_result_list = []
        r2_result_list = []

        for n_i in tqdm(keep_idxs, desc='co-bps'):
            bps = bits_per_spike(
                pred[:, :, [n_i]],
                gt_held_out[:, :, [n_i]],
            )
            _r2_list = []
            for k in range(pred.shape[0]):
                r2 = r2_score_sklearn(
                    gt_held_out[k, :, n_i],
                    pred[k, :, n_i],
                )
                _r2_list.append(r2)

            r2 = np.nanmean(_r2_list)
            r2_result_list.append(r2)
            if np.isinf(bps):
                bps = np.nan
            bps_result_list.append(bps)
        co_bps = np.nanmean(bps_result_list)
        print(f"Co-BPS: {co_bps}")
        print(f"r2: {np.nanmean(r2_result_list)}")
        test_bps.append(co_bps)

        result[eid] = {
            'gt': gt_held_out,
            'pred': pred,
            'bps': bps_result_list,
            'r2': r2_result_list,
            'eid': eid,
        }
    return result

            