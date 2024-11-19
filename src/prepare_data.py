import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict
from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior,
    load_video,
    load_video_index,
    get_whisker_pad_roi,
    load_whisker_video,
    get_optic_flow,
    load_behavior
)
import json
import webdataset as wds
from tqdm import tqdm
import tarfile
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="/expanse/lustre/scratch/ywang74/temp_project/Downloads")
ap.add_argument("--datasets", type=str, default="reproducible_ephys", choices=["reproducible-ephys", "brain-wide-map"])
ap.add_argument("--huggingface_org", type=str, default="neurofm123")
ap.add_argument("--n_sessions", type=int, default=6)
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--eid", type=str)
args = ap.parse_args()

SEED = 42

np.random.seed(SEED)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True,
    cache_dir = args.base_path
)
dataset_name = 'ibl-video'
os.makedirs(os.path.join(args.base_path, dataset_name), exist_ok=True)
freeze_file = 'data/bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

if args.eid is not None: 
    include_eids = [args.eid]
else:
    if args.datasets == "brain-wide-map":
        n_sub = args.n_sessions
        subjects = np.unique(bwm_df.subject)
        selected_subs = np.random.choice(subjects, n_sub, replace=False)
        by_subject = bwm_df.groupby('subject')
        include_eids = np.array([bwm_df.eid[by_subject.groups[sub][0]] for sub in selected_subs])
    else:
        with open('data/eid.txt') as file:
            include_eids = [line.rstrip() for line in file]
        include_eids = include_eids[:args.n_sessions]

# Trial setup
params = {
    'interval_len': 2, 
    'binsize': 0.02, 
    'single_region': False,
    'align_time': 'stimOn_times', 
    'time_window': (-.5, 1.5), 
    'fr_thresh': 0.5
}

beh_names = [
    'choice', 'reward', 'block', 
    'wheel-speed', 'whisker-motion-energy', #'body-motion-energy', 
    # 'dlc-pupil-bottom-r-y','dlc-pupil-top-r-y', 'dlc-pupil-left-r-x', 'dlc-pupil-right-r-x',
    #'pupil-diameter', # Some sessions do not have pupil traces
]
camera = 'left'

for eid_idx, eid in enumerate(include_eids):

    # try: 
    print('==========================')
    print(f'Preprocess session {eid}:')
    
    # Load and preprocess AP and behavior
    neural_dict, behave_dict, meta_data, trials_data, _ = prepare_data(
        one, eid, bwm_df, params, n_workers=args.n_workers
    )

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
    binned_spikes, clusters_used_in_bins, intervals = bin_spiking_data(
        region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=args.n_workers, **params
    )

    # load video index list
    video_index_list, url = load_video_index(one, eid, camera, intervals)
    # get whisker pad roi
    roi, mask = get_whisker_pad_roi(one, eid, camera)
    

    avg_fr = binned_spikes.sum(1).mean(0) / params['interval_len']
    active_neuron_ids = np.argwhere(avg_fr > 1/params['fr_thresh']).flatten()
    binned_spikes = binned_spikes[:,:,active_neuron_ids]
    print(f'# of neurons left after filtering out inactive ones: {binned_spikes.shape[-1]}/{len(avg_fr)}.')
  
    binned_behaviors, behavior_masks = bin_behaviors(
        one, 
        eid, 
        behaviors=beh_names[3:], 
        trials_df=trials_data['trials_df'], 
        allow_nans=True, 
        n_workers=args.n_workers, 
        freq=60, # Hz number of bins per second
        **params
    )

    print(f'binned_spikes: {binned_spikes.shape}')
    print(f'binned_behaviors: {binned_behaviors["wheel-speed"].shape}')
    # print(f'binned_lfp: {binned_lfp.shape}')
    
    print(beh_names)
    # Ensure neural and behavior data match for each trial
    aligned_binned_spikes, aligned_binned_behaviors, _, _ = align_spike_behavior(
        binned_spikes, binned_behaviors, beh_names, trials_data['trials_mask']
    )

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
    max_num_trials = len(aligned_binned_spikes)
    trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
    train_idxs = trial_idxs[:int(0.7*max_num_trials)]
    val_idxs = trial_idxs[int(0.7*max_num_trials):int(0.8*max_num_trials)]
    test_idxs = trial_idxs[int(0.8*max_num_trials):]

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in aligned_binned_behaviors.keys():
        train_beh.update({beh: aligned_binned_behaviors[beh][train_idxs]})
        val_beh.update({beh: aligned_binned_behaviors[beh][val_idxs]})
        test_beh.update({beh: aligned_binned_behaviors[beh][test_idxs]})
    
    for trial_id in tqdm(range(max_num_trials)):
        # trial spike
        spike = aligned_binned_spikes[trial_id]
        # trial behavior
        beh = {key: aligned_binned_behaviors[key][trial_id] for key in beh_names}
        # beh['whisker-motion-energy'] = load_behavior(one, eid, 'whisker-motion-energy', video_index_list[trial_id])
        # trial video
        _trial_video = load_video(video_index_list[trial_id], url)
        # load whisker video
        whisker_video = load_whisker_video(video_index_list[trial_id], url, mask)
        # check shape
        # assert spike.shape[0] == beh['wheel-speed'].shape[0]
        eid = meta_data['eid']
        sample_freq = meta_data['sampling_freq']
        cluster_channels = meta_data['cluster_channels']
        cluster_regions = meta_data['cluster_regions']
        good_clusters = meta_data['good_clusters']
        cluster_depths = meta_data['cluster_depths']
        trial_meta = {
            'eid': eid,
            'trial_id': trial_id,
            'sample_freq': sample_freq,
            'cluster_channels': cluster_channels,
            'cluster_regions': cluster_regions,
            'good_clusters': good_clusters,
            'cluster_depths': cluster_depths,
            'frame_time_idx': video_index_list[trial_id].tolist(),
            'interval': intervals[trial_id].tolist(),
            'roi': roi.tolist(),
            **params
        }
        whisker_of = get_optic_flow(
            video=whisker_video, 
            save_path=f'{eid[:5]}_of.mp4',
            ses=eid[:5],
            trial=trial_id,
        )
        exit()

        # add prefix to keys
        whisker_of = {f'whisker-{key}': value for key, value in whisker_of.items()}
        out_video = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (128, 128), isColor=False)
        trial_video = []
        for i in range(_trial_video.shape[0]):
            frame = cv2.resize(_trial_video[i], (128, 128))
            out_video.write(frame)
            trial_video.append(frame)
        out_video.release()
        trial_video = np.array(trial_video)

        out_whisker_video = cv2.VideoWriter('whisker_temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (whisker_video.shape[2], whisker_video.shape[1]), isColor=False)
        for i in range(whisker_video.shape[0]):
            out_whisker_video.write(whisker_video[i])
        out_whisker_video.release()
        
        whole_of = get_optic_flow(
            video=trial_video, 
            save_path=None,
            ses=eid[:5],
            trial=trial_id
        )
        whole_of = {f'whole-{key}': value for key, value in whole_of.items()}

        trial_data = {
            'ap': spike,
            **whisker_of,
            **whole_of,
            **beh
        }
        # each key in trial_data add .pyd
        trial_data = {key + '.pyd': value for key, value in trial_data.items()}
        sample_key = f'{eid}_{trial_id}'
        sample_dict = {
            '__key__': sample_key,
            **trial_data,
            'meta.json': json.dumps(trial_meta)
            }

        sink_path = os.path.join(args.base_path, dataset_name, f'{eid}_{trial_id}')
        with wds.TarWriter(sink_path + '.tar') as sink:
            sink.write(sample_dict)

        # add video to the tar file
        with tarfile.open(sink_path + '.tar', 'a') as tar:
            tar.add('temp.mp4', arcname=f'{sample_key}.video.mp4')
            tar.add('whisker_temp.mp4', arcname=f'{sample_key}.whisker-video.mp4')
        os.remove('temp.mp4')
        os.remove('whisker_temp.mp4')

print('Done!')
