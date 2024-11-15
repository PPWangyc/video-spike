from utils.utils import (
    get_args,
    set_seed,
    NAME2MODEL
)
from model.rrr import (
    train_model,
    train_model_main
)
from utils.dataset_utils import (
    split_dataset,
    get_metadata_from_loader
)
from utils.utils import (
    _std
)
from utils.metric_utils import (
    bits_per_spike,
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from loader.make import (
    make_loader
)
from trainer.make import (
    make_base_trainer
)
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.metrics import r2_score 
from tqdm import tqdm

def main():
    # set config
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)

    # select 100 idx from 120
    idx = np.random.choice(119, 100, replace=False)
    sorted_idx = np.sort(idx)

    # set seed
    set_seed(config.seed)
    # set dataset
    dataset_split_dict = split_dataset(config.dirs.data_dir,eid=args.eid)
    train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
    meta_data = get_metadata_from_loader(train_dataloader, config)
    print(f"meta_data: {meta_data}")
    eid = args.eid
    train_data = {
        eid:
        {
            "X": [], 
            "y": [], 
            "setup": {}
        }
    }
    smooth_w = 2; T = 120
    train_X = []; train_y = []
    input_mod = None
    if args.input_mod =='me':
        input_mod = 'whisker-motion-energy'
    elif args.input_mod == 'of':
        input_mod = 'whisker-of'
    elif args.input_mod == 'of-2d':
        input_mod = 'whisker-of-2d'
    elif args.input_mod == 'of-2d-v':
        input_mod = 'whisker-of-video'
    
    for batch in train_dataloader:
        if input_mod == 'whisker-of-video':
            x_vec = torch.tensor(np.median(batch[input_mod][...,0].numpy(),axis=(2,3)))
            y_vec = torch.tensor(np.median(batch[input_mod][...,1].numpy(),axis=(2,3)))
            # x_vec = batch[input_mod][...,0].mean(axis=(2,3))
            # y_vec = batch[input_mod][...,1].mean(axis=(2,3))
            value = torch.stack([x_vec, y_vec], dim=2)
            # took median of x and y
            
            # print(value.shape)
            # exit()
            train_X.append(value.numpy())
        else:
            train_X.append(batch[input_mod].numpy())
        train_y.append(batch["ap"].numpy())
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    train_y = gaussian_filter1d(train_y, smooth_w, axis=1)
    # if args.input_mod == 'of-2d-v':
    #     # normalize x and y
    #     train_X[...,0] = (train_X[...,0] - np.min(train_X[...,0])) / (np.max(train_X[...,0]) - np.min(train_X[...,0]))
    #     train_X[...,1] = (train_X[...,1] - np.min(train_X[...,1])) / (np.max(train_X[...,1]) - np.min(train_X[...,1]))

    val_X = []; val_y = []
    for batch in test_dataloader:
        if input_mod == 'whisker-of-video':
            x_vec = torch.tensor(np.median(batch[input_mod][...,0].numpy(),axis=(2,3)))
            y_vec = torch.tensor(np.median(batch[input_mod][...,1].numpy(),axis=(2,3)))
            # x_vec = batch[input_mod][...,0].mean(axis=(2,3))
            # y_vec = batch[input_mod][...,1].mean(axis=(2,3))
            value = torch.stack([x_vec, y_vec], dim=2)
            # value = batch[input_mod].mean(axis=(2,3,4))
            val_X.append(value.numpy())
        else:
            val_X.append(batch[input_mod].numpy())
        val_y.append(batch["ap"].numpy())
    val_X = np.concatenate(val_X, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    val_y = gaussian_filter1d(val_y, smooth_w, axis=1)
    # if args.input_mod == 'of-2d-v':
    #     # normalize x and y
    #     val_X[...,0] = (val_X[...,0] - np.min(val_X[...,0])) / (np.max(val_X[...,0]) - np.min(val_X[...,0]))
    #     val_X[...,1] = (val_X[...,1] - np.min(val_X[...,1])) / (np.max(val_X[...,1]) - np.min(val_X[...,1]))

    train_data[eid]["X"].append(train_X)
    train_data[eid]["y"].append(train_y)
    train_data[eid]["X"].append(val_X)
    train_data[eid]["y"].append(val_y)


    _, mean_X, std_X = _std(train_data[eid]["X"][0])
    _, mean_y, std_y = _std(train_data[eid]["y"][0])
    print(len(train_data[eid]["X"]), len(train_data[eid]["y"]))
    
    for i in range(2):
        K = train_data[eid]["X"][i].shape[0]
        T = train_data[eid]["X"][i].shape[1]
        # 
        train_data[eid]["X"][i] = (train_data[eid]["X"][i]-mean_X)/std_X
        
        if len(train_data[eid]["X"][i].shape) ==2:
            train_data[eid]["X"][i] = torch.tensor(train_data[eid]["X"][i]).unsqueeze(2).numpy()
        print(train_data[eid]["X"][i].shape)
        train_data[eid]["X"][i] = np.concatenate(
            [
                train_data[eid]["X"][i],
                np.ones((K, T, 1))
                ],
            axis=2
        )
        train_data[eid]["X"][i] = train_data[eid]["X"][i][:,sorted_idx]
        train_data[eid]["y"][i] = (train_data[eid]["y"][i]-mean_y)/std_y

    train_data[eid]["setup"]["mean_X_Tv"] = mean_X
    train_data[eid]["setup"]["std_X_Tv"] = std_X
    train_data[eid]["setup"]["mean_y_TN"] = mean_y
    train_data[eid]["setup"]["std_y_TN"] = std_y
    
    l2 = 100
    n_comp = 3
    print('start training')
    model, mse_val = train_model_main(
        train_data=train_data,
        l2=l2,
        n_comp=n_comp,
        model_fname='tmp',
        save=True
    )
    print('finished training')
    _, _, pred_orig = model.predict_y_fr(train_data, eid, 1)
    pred_orig = pred_orig.cpu().detach().numpy()
    print(pred_orig.shape)
    
    
    # ----
    # TEST
    # ----
    trial_len = 2.
    threshold = 1e-3
    pred_held_out = np.clip(pred_orig, threshold, None)
    gt_held_out = []
    for batch in test_dataloader:
        gt_held_out.append(batch["ap"].numpy())
    gt_held_out = np.concatenate(gt_held_out, axis=0)
    mean_fr = gt_held_out.sum(1).mean(0) / trial_len
    keep_idxs = np.arange(len(mean_fr)).flatten()
    print(keep_idxs)
    print(mean_fr.shape)

    print(pred_held_out.shape, gt_held_out.shape)

    bps_result_list = []
    r2_result_list = []

    for n_i in tqdm(keep_idxs, desc='co-bps'):
        bps = bits_per_spike(
            pred_held_out[:, :, n_i],
            gt_held_out[:, :, n_i]        
        )
        _r2_list = []
        for k in range(pred_held_out.shape[0]):
            r2 = r2_score(
                gt_held_out[k, :, n_i],
                pred_held_out[k, :, n_i]
            )
            _r2_list.append(r2)
        r2 = np.nanmean(_r2_list)
        r2_result_list.append(r2)
        if np.isinf(bps):
            bps = np.nan
        bps_result_list.append(bps)
    print(f"co-bps: {np.nanmean(bps_result_list)}")
    print(f"population bps: {bits_per_spike(pred_held_out, gt_held_out)}")
    print(f"r2: {np.nanmean(r2_result_list)}")
    
    result = {
        'gt': gt_held_out,
        'pred': pred_held_out,
        'co_bps': bps_result_list,
        'r2': r2_result_list,
        'eid': eid,
    }
    np.save(f'{eid[:5]}_{args.input_mod}_result.npy', result)
    
    

if __name__ == '__main__':
    main()
