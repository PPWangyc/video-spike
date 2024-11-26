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
    _std,
    _one_hot
)
from utils.metric_utils import (
    bits_per_spike,
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)

import torch
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

    # set seed
    set_seed(config.seed)

    with open('data/eid.txt') as file:
        include_eids = [line.rstrip() for line in file]
        # include_eids = include_eids[:args.n_sessions]
    
    # select 100 idx from 120
    idx = np.random.choice(119, 100, replace=False)
    sorted_idx = np.sort(idx)
    
    input_mod = None
    if args.input_mod =='me':
        input_mod = 'me'
    elif args.input_mod == 'of':
        input_mod = 'whisker-of'
    elif args.input_mod == 'of-2d':
        input_mod = 'of-2d'
    elif args.input_mod == 'of-2d-v':
        input_mod = 'whisker-of-video'
    elif args.input_mod == 'all':
        input_mod = 'all'
    elif args.input_mod == 'other':
        input_mod = 'other'
    elif args.input_mod == 'me-all':
        input_mod = 'all'
    elif args.input_mod == 'of-all':
        input_mod = 'of-all'
    elif args.input_mod == 'cebra':
        input_mod = 'cebra'
    elif args.input_mod == 'pca':
        input_mod = 'pca'
    
    # set dataset
    train_data = np.load(f'data/data_rrr_{input_mod}.npy', allow_pickle=True).item()
    smooth_w = 2; T = 100
    ground_truth = {}
    eids = train_data.keys()
    # sort the eids
    eids = sorted(eids)
    # apply gaussian filteer to the ground truth
    # one-hot encoding for choice and block
    for eid in eids:
        ground_truth[eid] = train_data[eid]["y"][1]
        for i in range(2):
            train_data[eid]["y"][i] = gaussian_filter1d(train_data[eid]["y"][i], smooth_w, axis=1)
            if args.input_mod =='cebra' or args.input_mod == 'pca':
                print(train_data[eid]["X"][i].shape)
                continue
            # one-hot encoding for choice and block
            if args.input_mod != 'me' and args.input_mod != 'of-2d':
                input = train_data[eid]["X"][i]
                choice = input[:,0,-2:-1]
                block = input[:,0,-1:]
                if args.input_mod == 'me-all' or args.input_mod == 'of-all':
                    const = 3
                else:
                    const = 2
                contin_dim = input.shape[2]-const
                choice = _one_hot(choice,120)
                block = _one_hot(block,120)
                input = np.concatenate([choice, block,input[...,-2-contin_dim:-2]], axis=2)
                train_data[eid]["X"][i] = input

    for eid in eids:
        _, mean_X, std_X = _std(train_data[eid]["X"][0])
        _, mean_y, std_y = _std(train_data[eid]["y"][0])

        for i in range(2):
            K = train_data[eid]["X"][i].shape[0]
            T = train_data[eid]["X"][i].shape[1]

            train_data[eid]["X"][i] = (train_data[eid]["X"][i]-mean_X)/std_X
            
            if len(train_data[eid]["X"][i].shape) ==2:
                train_data[eid]["X"][i] = torch.tensor(train_data[eid]["X"][i]).unsqueeze(2).numpy()
            print(train_data[eid]["X"][i].shape, train_data[eid]["y"][i].shape)
            # add bias term
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
    
    result = {}
    test_bps = []
    for eid in eids:
        if '03d9a09' in eid:
            continue
        _train_data = {eid:train_data[eid]}
        model, mse_val = train_model_main(
            train_data=_train_data,
            l2=l2,
            n_comp=n_comp,
            model_fname='tmp',
            save=True
        )
        print('finished training')
        # ----
        # TEST
        # ----
        print('eid:', eid)
        _, _, pred_orig = model.predict_y_fr(train_data, eid, 1)
        pred = pred_orig.cpu().detach().numpy()
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
                gt_held_out[:, :, [n_i]]        
            )
            _r2_list = []
            for k in range(pred.shape[0]):
                r2 = r2_score(
                    gt_held_out[k, :, n_i],
                    pred[k, :, n_i]
                )
                _r2_list.append(r2)
            r2 = np.nanmean(_r2_list)
            r2_result_list.append(r2)
            if np.isinf(bps):
                bps = np.nan
            bps_result_list.append(bps)
        co_bps = np.nanmean(bps_result_list)
        print(f"co-bps: {co_bps}")
        # print(f"population bps: {bits_per_spike(pred, gt_held_out)}")
        print(f"r2: {np.nanmean(r2_result_list)}")
        test_bps.append(co_bps)

        result[eid] = {
            'gt': gt_held_out,
            'pred': pred,
            'co_bps': bps_result_list,
            'r2': r2_result_list,
            'eid': eid,
        }
    print(result.keys())
    for i in range(len(test_bps)):
        print(f'{test_bps[i]:.5f}')
    print(f'mean bps:{np.mean(test_bps):.5f}')
    print(f"Total num of eid: {len(result.keys())}")
    np.save(f'{args.input_mod}_result.npy', result)
    
    
    

if __name__ == '__main__':
    main()
