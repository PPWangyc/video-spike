import os 
import sys
import uuid
from tqdm import *
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from functools import partial
from scipy.interpolate import interp1d
import cv2
import imageio
from PIL import Image

import ibllib.io.video as vidio
from iblutil.numerical import ismember, bincount2D
import brainbox.behavior.dlc as dlc
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.regions import BrainRegions
from brainbox.population.decode import get_spike_counts_in_bins

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter


def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result


def load_spiking_data(one, pid, compute_metrics=False, qc=None, **kwargs):
    """
    Function to load the cluster information and spike trains for clusters that may or may not pass certain quality metrics.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    pid: str
        A probe insertion UUID
    compute_metrics: bool
        If True, force SpikeSortingLoader.merge_clusters to recompute the cluster metrics. Default is False
    qc: float
        Quality threshold to be used to select good clusters. Default is None.
        If use all available clusters, set qc to None. If use good clusters, set qc to 1.
    kwargs:
        Keyword arguments passed to SpikeSortingLoader upon initiation. Specifically, if one instance offline,
        you need to pass 'eid' and 'pname' here as they cannot be inferred from pid in offline mode.

    Returns
    -------
    selected_spikes: dict
        Spike trains associated with clusters. Dictionary with keys ['depths', 'times', 'clusters', 'amps']
    selected_clusters: pandas.DataFrame
        Information of clusters for this pid 
    sampling_freq: float
        Sampling frequency of spiking data (action potential)
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    sampling_freq = spike_loader.raw_electrophysiology(band="ap", stream=True).fs
    
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics).to_df()
    if qc is None:
        return spikes, clusters_labeled, sampling_freq
    else:
        iok = clusters_labeled['label'] >= qc
        selected_clusters = clusters_labeled[iok]
        spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
        selected_clusters.reset_index(drop=True, inplace=True)
        selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
        selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
        return selected_spikes, selected_clusters, sampling_freq


def merge_probes(spikes_list, clusters_list):
    """
    Merge spikes and clusters information from several probes as if they were recorded from the same probe.
    This can be used to account for the fact that data from the probes recorded in the same session are not
    statistically independent as they have the same underlying behaviour.

    NOTE: The clusters dataframe will be re-indexed to avoid duplicated indices. Accordingly, spikes['clusters']
    will be updated. To unambiguously identify clusters use the column 'uuids'

    Parameters
    ----------
    spikes_list: list of dicts
        List of spike dictionaries as loaded by SpikeSortingLoader or brainwidemap.load_good_units
    clusters_list: list of pandas.DataFrames
        List of cluster dataframes as loaded by SpikeSortingLoader.merge_clusters or brainwidemap.load_good_units

    Returns
    -------
    merged_spikes: dict
        Merged and time-sorted spikes in single dictionary, where 'clusters' is adjusted to index into merged_clusters
    merged_clusters: pandas.DataFrame
        Merged clusters in single dataframe, re-indexed to avoid duplicate indices.
        To unambiguously identify clusters use the column 'uuids'
    """

    assert (len(clusters_list) == len(spikes_list)), 'clusters_list and spikes_list must have the same length'
    assert all([isinstance(s, dict) for s in spikes_list]), 'spikes_list must contain only dictionaries'
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), 'clusters_list must contain only pd.DataFrames'

    merged_spikes = []
    merged_clusters = []
    cluster_max = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes['clusters'] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
        
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()}
    # Sort spikes by spike time
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}

    return merged_spikes, merged_clusters


#def load_trials_and_mask(
#        one, eid, min_rt=None, max_rt=None, nan_exclude='default', min_trial_len=None,
#        max_trial_len=None, exclude_unbiased=False, exclude_nochoice=True, sess_loader=None):
def load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', min_trial_len=None,
        max_trial_len=10, exclude_unbiased=False, exclude_nochoice=True, sess_loader=None):
    """
    Function to load all trials for a given session and create a mask to exclude all trials that have a reaction time
    shorter than min_rt or longer than max_rt or that have NaN for one of the specified events.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to local or remote database
    eid: str
        A session UUID
    min_rt: float or None
        Minimum admissible reaction time in seconds for a trial to be included. Default is 0.08. If None, don't apply.
    max_rt: float or None
        Maximum admissible reaction time in seconds for a trial to be included. Default is 2. If None, don't apply.
    nan_exclude: list or 'default'
        List of trial events that cannot be NaN for a trial to be included. If set to 'default' the list is
        ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']
    min_trial_len: float or None
        Minimum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    max_trial_len: float or Nona
        Maximum admissible trial length measured by goCue_time (start) and feedback_time (end).
        Default is None.
    exclude_unbiased: bool
        True to exclude trials that fall within the unbiased block at the beginning of session.
        Default is False.
    exclude_nochoice: bool
        True to exclude trials where the animal does not respond. Default is False.
    sess_loader: brainbox.io.one.SessionLoader or NoneType
        Optional SessionLoader object; if None, this object will be created internally

    Returns
    -------
    trials: pandas.DataFrame
        Trials table containing all trials for this session. If complete with columns:
        ['stimOff_times','goCueTrigger_times','feedbackType','contrastLeft','contrastRight','rewardVolume',
        'goCue_times','choice','feedback_times','stimOn_times','response_times','firstMovement_times',
        'probabilityLeft', 'intervals_0', 'intervals_1']
    mask: pandas.Series
        Boolean Series to mask trials table for trials that pass specified criteria. True for all trials that should be
        included, False for all trials that should be excluded.
    """

    if nan_exclude == 'default':
        nan_exclude = [
            'stimOn_times',
            'choice',
            'feedback_times',
            'probabilityLeft',
            'firstMovement_times',
            'feedbackType'
        ]
    
    if sess_loader is None:
        sess_loader = SessionLoader(one, eid=eid)

    if sess_loader.trials.empty:
        # print(sess_loader)
        sess_loader.load_trials()

    # Create a mask for trials to exclude
    # Remove trials that are outside the allowed reaction time range
    if min_rt is not None:
        query = f'(firstMovement_times - stimOn_times < {min_rt})'
    else:
        query = ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    # Remove trials that are outside the allowed trial duration range
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    # Remove trials with nan in specified events
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    # Remove trials in unbiased block at beginning
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    # Remove trials where animal does not respond
    if exclude_nochoice:
        query += ' | (choice == 0)'
    # If min_rt was None we have to clean up the string
    if min_rt is None:
        query = query[3:]

    # Create mask
    mask = ~sess_loader.trials.eval(query)

    return sess_loader.trials, mask


def list_brain_regions(neural_dict, **kwargs):
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')
    regions = ([[k] for k in np.unique(beryl_reg)] if kwargs['single_region'] else [np.unique(beryl_reg)])
    print(f"Use spikes from brain regions: ", regions[0])
    return regions, beryl_reg


def select_brain_regions(regressors, beryl_reg, region, **kwargs):
    """
    Select units based on brain region.
    """
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def create_intervals(start_time, end_time, interval_len):
    interval_begs = np.arange(
        start_time, end_time-interval_len, interval_len
    )
    interval_ends = np.arange(
        start_time+interval_len, end_time, interval_len
    )
    return np.c_[interval_begs, interval_ends]


def get_spike_data_per_interval(times, clusters, interval_begs, interval_ends, interval_len, binsize, n_workers=os.cpu_count()):
    """
    Select spiking data for specified interval in each recording.

    Parameters
    ----------
    times : array-like
        time in seconds for each spike
    clusters : array-like
        cluster id for each spike
    interval_begs : array-like
        beginning of each interval in seconds
    interval_ends : array-like
        end of each interval in seconds
    interval_len : float
    binsize : float
        width of each bin in seconds

    Returns
    -------
    tuple
        - (list): time in seconds for each interval; timepoints refer to the start/left edge of a bin
        - (list): data for each interval of shape (n_clusters, n_bins)

    """
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    # This allows multiprocessing to work with nested functions
    @globalize
    def compute_spike_count(interval):
        # We use interval_idx to track the interval order while working with p.imap_unordered()
        interval_idx, t_beg, t_end = interval
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        return binned_spikes_tmp[:, :n_bins], idxs_tmp, interval_idx

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))
    with multiprocessing.Pool(processes=n_workers) as p:
        intervals = list(zip(np.arange(n_intervals), interval_begs, interval_ends))
        with tqdm(total=len(intervals)) as pbar:
            for res in p.imap_unordered(compute_spike_count, intervals):
                pbar.update()
                binned_spikes[res[-1], res[1], :] += res[0]
        pbar.close()
        p.close()
    return binned_spikes


def bin_spiking_data(reg_clu_ids, neural_df, intervals=None, trials_df=None, n_workers=os.cpu_count(), **kwargs):
    """
    Format a single session-wide array of spikes into a list of trial-based arrays.
    The ordering of clusters used in the output are also returned.

    Parameters
    ----------
    reg_clu_ids : array-like
        array of cluster ids for each spike
    neural_df : pd.DataFrame
        keys: 'spike_times', 'spike_clusters', 'cluster_regions', 'cluster_qc', 'cluster_df'
    intervals : 
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        columns: 'choice', 'feedback', 'pLeft', 'firstMovement_times', 'stimOn_times',
        'feedback_times'
    kwargs
        align_time : str
            event in trial on which to align intervals
            'firstMovement_times' | 'stimOn_times' | 'feedback_times'
        time_window : tuple
            (window_start, window_end), relative to align_time
        binsize : float, optional
            size of bins in seconds for multi-bin decoding
            
    Returns
    -------
    list
        each element is a 2D numpy.ndarray for a single interval of shape (n_bins, n_clusters)
    array
        cluster ids that account for axis 1 of the above 2D arrays.
    """

    if trials_df is not None:
        # compute time intervals for each trial
        intervals = np.vstack([
            trials_df[kwargs['align_time']] + kwargs['time_window'][0],
            trials_df[kwargs['align_time']] + kwargs['time_window'][1]
        ]).T
        chunk_len = kwargs['time_window'][1] - kwargs['time_window'][0]
        interval_len = (
            kwargs['time_window'][1] - kwargs['time_window'][0]
        )
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        chunk_len = intervals[0,1] - intervals[0,0]
        interval_len = (
            intervals[0,1] - intervals[0,0]
        )

    # subselect spikes for this region
    spikemask = np.isin(neural_df['spike_clusters'], reg_clu_ids)
    regspikes = neural_df['spike_times'][spikemask]
    regclu = neural_df['spike_clusters'][spikemask]
    clusters_used_in_bins = np.unique(regclu)

    binsize = kwargs.get('binsize', chunk_len)
    
    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs['binsize'],
            n_workers=n_workers)
        binned_list = [x.T for x in binned_array]   
    return np.array(binned_list), clusters_used_in_bins, intervals

def load_dlc_data(one, eid, camera='left',collection='alf'):
    """
    Load DLC data from a specific camera for a given session.

    Parameters
    ----------
    one : 
    eid : str
    camera : str
        'leftCamera' | 'rightCamera'
    attribute : list
        e.g., ['dlc', 'features', 'times']
    collection : str, optional
        'alf' | 'raw_video_frames' | 'raw_video_data'

    Returns
    -------
    dict
        'times': timestamps for DLC data
        'values': associated values
    """
    attribute = ['dlc', 'features', 'times']
    camera=f'{camera}Camera'
    dlc_data = one.load_object(eid, camera, attribute=attribute, collection=collection)
    return dlc_data
    
def load_target_behavior(one, eid, target):
    """
    Parameters
    ----------
    target : str
        'wheel-position' | 'wheel-velocity' | 'wheel-speed' | 
        'left-whisker-motion-energy' | 'right-whisker-motion-energy' | 
        'left-pupil-diameter' | 'right-pupil-diameter' |
        'left-camera-left-paw-speed' | 'left-camera-right-paw-speed' | 
        'right-camera-left-paw-speed' | 'right-camera-right-paw-speed' |
        'left-nose-speed' | 'right-nose-speed'
    one : 
    eid : str

    Returns
    -------
    dict
        'times': timestamps for behavior signal
        'values': associated values
        'skip': bool, True if there was an error loading data
    """

    # To load wheel and motion energy, we just use the SessionLoader, e.g.
    sess_loader = SessionLoader(one, eid=eid)
    
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    try:
        if target == 'wheel-position':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['position'].to_numpy()
            }
        elif target == 'wheel-velocity':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': sess_loader.wheel['velocity'].to_numpy()
            }
        elif target == 'wheel-speed':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy())
            }
    
        # motion_energy is a dictionary of dataframes, each containing the times and the motion energy for each view
        # for the side views, they contain columns ['times', 'whiskerMotionEnergy'] for the body view it contains
        # ['times', 'bodyMotionEnergy']
        elif target == 'left-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['left'])
            beh_dict = {
                'times': sess_loader.motion_energy['leftCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy'].to_numpy()
            }
        elif target == 'right-whisker-motion-energy':
            sess_loader.load_motion_energy(views=['right'])
            beh_dict = {
                'times': sess_loader.motion_energy['rightCamera']['times'].to_numpy(),
                'values': sess_loader.motion_energy['rightCamera']['whiskerMotionEnergy'].to_numpy()
            }
    
        # To load pose (DLC) data, e.g.
        # TO DO: Add pupil traces from lightning pose when they become available in the IBL database, e.g.,
        #        sessions = one.search(dataset='lightningPose', details=False)
        #        pupil_data = one.load_object(eid, f'leftCamera', attribute=['lightningPose', 'times'])
        # TO DO: Sometimes some traces are unavailable. Right now we still load them as 'nan' but need to handle it later.
        # TO DO: Different cameras have very different traces for the same behavior. Treat them as independent? 
        elif target == 'left-pupil-diameter':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc_left.features.pupilDiameter_smooth
            }
        elif target == 'right-pupil-diameter':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc_right.features.pupilDiameter_smooth
            }
        elif target == 'dlc-pupil-bottom-r-y':
            lp_left = one.load_object(eid, f'rightCamera', collection="alf")
            beh_dict = {
                'times': lp_left['times'],
                'values': lp_left['dlc']['pupil_bottom_r_y']
            }
        elif target == 'dlc-pupil-top-r-y':
            lp_left = one.load_object(eid, f'rightCamera', collection="alf")
            beh_dict = {
                'times': lp_left['times'],
                'values': lp_left['dlc']['pupil_top_r_y']
            }
        elif target == 'dlc-pupil-left-r-x':
            lp_left = one.load_object(eid, f'rightCamera', collection="alf")
            beh_dict = {
                'times': lp_left['times'],
                'values': lp_left['dlc']['pupil_left_r_x']
            }
        elif target == 'dlc-pupil-right-r-x':
            lp_left = one.load_object(eid, f'rightCamera', collection="alf")
            beh_dict = {
                'times': lp_left['times'],
                'values': lp_left['dlc']['pupil_right_r_x']
            }
        elif target == 'lightning-pose-left-pupil-diameter':
            lp_left = one.load_object(eid, f'leftCamera', attribute=['lightningPose', 'times'])
            dm1 = np.fabs(
                lp_left['lightningPose']['pupil_right_r_x'] - \
                lp_left['lightningPose']['pupil_left_r_x']
            )
            dm2 = np.fabs(
                lp_left['lightningPose']['pupil_top_r_y'] - \
                lp_left['lightningPose']['pupil_bottom_r_y']
            )
            assert (np.allclose(dm1, dm2))
            beh_dict = {
                'times': lp_left['times'],
                'values': dm1
            }
        elif target == 'lightning-pose-right-pupil-diameter':
            lp_right = one.load_object(eid, f'rightCamera', attribute=['lightningPose', 'times'])
            dm1 = np.fabs(
                lp_right['lightningPose']['pupil_right_r_x'] - \
                lp_right['lightningPose']['pupil_left_r_x']
            )
            dm2 = np.fabs(
                lp_right['lightningPose']['pupil_top_r_y'] - \
                lp_right['lightningPose']['pupil_bottom_r_y']
            )
            assert (np.allclose(dm1, dm2))
            beh_dict = {
                'times': lp_right['times'],
                'values': dm1
            }
        elif target == 'left-camera-left-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_l")
            }
        elif target == 'left-camera-right-paw-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="paw_r")
            }
        elif target == 'right-camera-left-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_l")
            }
        elif target == 'right-camera-right-paw-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="paw_r")
            }
        elif target == 'left-nose-speed':
            dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_left.times,
                'values': dlc.get_speed(dlc_left.dlc, dlc_left.times, camera="left", feature="nose_tip")
            }
        elif target == 'right-nose-speed':
            dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection="alf")
            beh_dict = {
                'times': dlc_right.times,
                'values': dlc.get_speed(dlc_right.dlc, dlc_right.times, camera="right", feature="nose_tip")
            }
        else:
            raise NotImplementedError
    except BaseException as e:
        print('Error loading %s data' % target)
        print(e)
        beh_dict = {'times': None, 'values': None, 'skip': True}
 
    return beh_dict


def get_behavior_per_interval(
    target_times, 
    target_vals, 
    intervals=None, 
    trials_df=None, 
    allow_nans=False, 
    freq=60,
    n_workers=os.cpu_count(), 
    **kwargs
):
    """
    Format a single session-wide array of target data into a list of interval-based arrays.

    Note: the bin size of the returned data will only be equal to the input `binsize` if that value
    evenly divides `align_interval`; for example if `align_interval=(0, 0.2)` and `binsize=0.10`,
    then the returned data will have the correct binsize. If `align_interval=(0, 0.2)` and
    `binsize=0.06` then the returned data will not have the correct binsize.

    Parameters
    ----------
    target_times : array-like
        time in seconds for each sample
    target_vals : array-like
        data samples
    intervals : 
        array of time intervals for each recording chunk including trials and non-trials
    trials_df : pd.DataFrame
        requires a column that matches `align_event`
    align_event : str
        event to align interval to
        firstMovement_times | stimOn_times | feedback_times
    align_interval : tuple
        (align_begin, align_end); time in seconds relative to align_event
    binsize : float
        size of individual bins in interval
    allow_nans : bool, optional
        False to skip intervals with >0 NaN values in target data

    Returns
    -------
    tuple
        - (list): time in seconds for each interval
        - (list): data for each interval
        - (array-like): mask of good intervals (True) and bad intervals (False)

    """

    binsize = kwargs['binsize']

    if trials_df is not None:
        align_event = kwargs['align_time']
        align_interval = kwargs['time_window']
        interval_len = align_interval[1] - align_interval[0]
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        interval_begs, interval_ends = intervals.T

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print('interval times all nan')
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        target_times_list = []
        target_vals_list = []
        return target_times_list, target_vals_list, good_interval

    # np.ceil because we want to make sure our bins contain all data
    # n_bins = int(np.ceil(interval_len / binsize))
    n_bins = int(freq * interval_len)
    binsize = interval_len / n_bins

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # interpolate and store
    target_times_list = [None for _ in range(len(target_times_og_list))]
    target_vals_list = [None for _ in range(len(target_times_og_list))]
    good_interval = [None for _ in range(len(target_times_og_list))]
    skip_reasons = [None for _ in range(len(target_times_og_list))]

    @globalize
    def interpolate_behavior(target):
        # We use interval_idx to track the interval order while working with p.imap_unordered()
        interval_idx, target_time, target_vals = target

        is_good_interval, x_interp, y_interp = False, None, None
        
        if len(target_vals) == 0:
            skip_reason = 'target data not present'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.sum(np.isnan(target_vals)) > 0 and not allow_nans:
            skip_reason = 'nans in target data'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
            skip_reason = 'bad interval data'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_begs[interval_idx] - target_time[0]) > binsize:
            skip_reason = 'target data starts too late'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason
        if np.abs(interval_ends[interval_idx] - target_time[-1]) > binsize:
            skip_reason = 'target data ends too early'
            return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

        is_good_interval, skip_reason = True, None
        x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)
        if len(target_vals.shape) > 1 and target_vals.shape[1] > 1:
            n_dims = target_vals.shape[1]
            y_interp_tmps = []
            for n in range(n_dims):
                y_interp_tmps.append(interp1d(
                    target_time, target_vals[:, n], kind='linear',
                    fill_value='extrapolate')(x_interp))
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)
        return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

    with multiprocessing.Pool(processes=n_workers) as p:
        targets = list(zip(np.arange(n_intervals), target_times_og_list, target_vals_og_list))
        with tqdm(total=n_intervals) as pbar:
            for res in p.imap_unordered(interpolate_behavior, targets):
                pbar.update()
                good_interval[res[0]] = res[1]
                target_times_list[res[0]] = res[2]
                target_vals_list[res[0]] = res[3]
                skip_reasons[res[0]] = res[-1]
        pbar.close()
        p.close()

    return target_times_list, target_vals_list, np.array(good_interval), skip_reasons    


def load_anytime_behaviors(one, eid, n_workers=os.cpu_count()):

    behaviors = [
        #'wheel-position', 'wheel-velocity', 
        'wheel-speed',
        'left-whisker-motion-energy', 'right-whisker-motion-energy',
        #'left-pupil-diameter', 'right-pupil-diameter',
        #'lightning-pose-left-pupil-diameter', 'lightning-pose-right-pupil-diameter'
        # These behaviors are of bad quality - skip them for now
        # 'left-camera-left-paw-speed', 'left-camera-right-paw-speed', 
        # 'right-camera-left-paw-speed', 'right-camera-right-paw-speed',
        # 'left-nose-speed', 'right-nose-speed'
    ]

    @globalize
    def load_beh(beh):
        return beh, load_target_behavior(one, eid, beh)
    
    behave_dict = {}
    with multiprocessing.Pool(processes=n_workers) as p:
        with tqdm(total=len(behaviors)) as pbar:
            for res in p.imap_unordered(load_beh, behaviors):
                pbar.update()
                behave_dict.update({res[0]: res[1]})
        pbar.close()
        p.close()
    
    return behave_dict


def bin_behaviors(
    one, 
    eid, 
    intervals=None, 
    trials_only=False, 
    trials_df=None, 
    mask=None, 
    allow_nans=True, 
    n_workers=os.cpu_count(),
    behaviors=None,
    freq=60,
    **kwargs
):
    assert behaviors is not None, 'Require a list of behaviors to bin.'

    behave_dict, mask_dict = {}, {}
    
    if mask is not None:
        trials_df = trials_df[mask]

    if trials_df is not None:        
        choice = trials_df['choice'].to_numpy()
        block = trials_df['probabilityLeft'].to_numpy()
        reward = (trials_df['rewardVolume'] > 1).astype(int).to_numpy()
        contrast = np.c_[trials_df['contrastLeft'], trials_df['contrastRight']]
        contrast = (-1 * np.nan_to_num(contrast, 0)).sum(1)

        behave_dict.update(
            {'choice': choice, 'block': block, 'reward': reward, 'contrast': contrast}
        )
        behave_mask = np.ones(len(trials_df)) 
    else:
        assert intervals is not None, 'Require intervals to segment the recording into chunks including trials and non-trials.'
        behave_mask = np.ones(len(intervals)) 
        
    for beh in behaviors:
        if beh == 'whisker-motion-energy':
            target_dict = load_target_behavior(one, eid, 'left-whisker-motion-energy')
            if 'skip' in target_dict.keys():
                target_dict = load_target_behavior(one, eid, 'right-whisker-motion-energy')
        elif beh == 'pupil-diameter':
            target_dict = load_target_behavior(one, eid, 'lightning-pose-left-pupil-diameter')
            if 'skip' in target_dict.keys():
                target_dict = load_target_behavior(one, eid, 'lightning-pose-right-pupil-diameter')
        else:
            target_dict = load_target_behavior(one, eid, beh)
        target_times, target_vals = target_dict['times'], target_dict['values']
        target_times_list, target_vals_list, target_mask, skip_reasons = get_behavior_per_interval(
            target_times, 
            target_vals, 
            intervals=intervals, 
            trials_df=trials_df, 
            allow_nans=allow_nans, 
            n_workers=n_workers, 
            freq=freq,
            **kwargs
        )
        behave_dict.update({beh: np.array(target_vals_list, dtype=object)})
        mask_dict.update({beh: target_mask})
        behave_mask = np.logical_and(behave_mask, target_mask)

    if not allow_nans:
        for k, v in behave_dict.items():
            behave_dict[k] = behave_dict[beh][behave_mask]
    
    return behave_dict, mask_dict


def prepare_data(one, eid, bwm_df, params, n_workers=os.cpu_count()):
    
    # When merging probes we are interested in eids, not pids
    #idx = (bwm_df.eid.unique() == eid).argmax()
    #eid = bwm_df.eid.unique()[idx]
    #tmp_df = bwm_df.set_index(['eid', 'subject']).xs(eid, level='eid')
    #subject = tmp_df.index[0]
    #lab = tmp_df.lab.iloc[0]

    pids, probe_names = one.eid2pid(eid)  # Select all probes of this session
    #pids = tmp_df['pid'].to_list() 
    #probe_names = tmp_df['probe_name'].to_list()
    #print(probe_names)
    #print(pids)
    print(f"Merge {len(probe_names)} probes for session eid: {eid}")

    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters, sampling_freq = load_spiking_data(one, pid, eid=eid, pname=probe_name)
        tmp_clusters['pid'] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    _, good_trials_mask = load_trials_and_mask(one=one, eid=eid)

    trials_df, trials_mask = load_trials_and_mask(one=one, eid=eid, min_rt=None, max_rt=None, max_trial_len=None)
        
    behave_dict = load_anytime_behaviors(one, eid, n_workers=n_workers)
    
    neural_dict = {
        'spike_times': spikes['times'],
        'spike_clusters': spikes['clusters'],
        'cluster_regions': clusters['acronym'].to_numpy(),
    }
        
    meta_data = {
        #'subject': subject,
        'eid': eid,
        #'probe_name': probe_name,
        #'lab': lab,
        'sampling_freq': sampling_freq,
        'cluster_channels': list(clusters['channels']),
        'cluster_regions': list(clusters['acronym']),
        'good_clusters': list((clusters['label'] >= 1).astype(int)),
        'cluster_depths': list(clusters['depths']),
        'uuids':  list(clusters['uuids']),
        'cluster_qc': {k: np.asarray(v) for k, v in clusters.to_dict('list').items()},
        # 'cluster_df': clusters
    }

    trials_data = {
        'trials_df': trials_df,
        'trials_mask': trials_mask
    }

    return neural_dict, behave_dict, meta_data, trials_data, good_trials_mask


def align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask=None):


    target_mask = [1] * len(binned_spikes)
    for beh_name in beh_names:
        beh_mask = [1 if trial is not None else 0 for trial in binned_behaviors[beh_name]]
    target_mask = target_mask and beh_mask

    if trials_mask is not None:
        target_mask = target_mask and list(trials_mask.to_numpy().astype(int))

    del_idxs = np.argwhere(np.array(target_mask) == 0)

    aligned_binned_spikes = np.delete(binned_spikes, del_idxs, axis=0)

    aligned_binned_behaviors = {}
    for beh_name in beh_names:
        aligned_binned_behaviors.update({beh_name: np.delete(binned_behaviors[beh_name], del_idxs, axis=0)})
        aligned_binned_behaviors[beh_name] = np.array(
                [y for y in aligned_binned_behaviors[beh_name]], dtype=float
            ).reshape((aligned_binned_spikes.shape[0], -1)
        )
        if beh_name in [
            'wheel-speed', 'whisker-motion-energy', 
            #'pupil-diameter'
        ]:
            aligned_binned_behaviors[beh_name] = (aligned_binned_behaviors[beh_name] - np.min(aligned_binned_behaviors[beh_name])) / (np.max(aligned_binned_behaviors[beh_name]) - np.min(aligned_binned_behaviors[beh_name]))
        assert len(aligned_binned_spikes) == len(aligned_binned_behaviors[beh_name]), f'mismatch between spike shape {len(aligned_binned_spikes)} and {beh_name} shape {len(aligned_binned_behaviors[beh_name])}'

    return aligned_binned_spikes, aligned_binned_behaviors, target_mask, del_idxs

def load_video_index(one, eid, camera, intervals):
    # get the remote video URL from eid
    urls = vidio.url_from_eid(eid, one=one)
    url = urls[camera]  # URL for the left camera

    # Example 2: get the video label from a video file path or URL
    label = vidio.label_from_path(url)
    print(f'Using URL for the {label} camera')
    """
    The preload function will by default pre-allocate the memory before loading the frames,
    and will return the frames as a numpy array of the shape (l, h, w, 3), where l = the number of
    frame indices given.  The indices must be an iterable of positive integers.  Because the videos
    are in black and white the values of each color channel are identical.   Therefore to save on
    memory you can provide a slice that returns only one of the three channels for each frame.  The
    resulting shape will be (l, h, w).  NB: Any slice or boolean array may be provided which is
    useful for cropping to an ROI.
    """

    meta = vidio.get_video_meta(url, one=one)
    for k, v in meta.items():
        print(f'The video {k} = {v}')
    fps = meta['fps']
    # Example 6: load video timestamps
    ts = one.load_dataset(eid, f'_ibl_{label}Camera.times.npy', collection='alf')
    interval_len = intervals[0,1] - intervals[0,0]
    reg_frame_num = int(fps * interval_len)
    session_frames =[]
    trial_index_list = []
    for trial in intervals:
        # ts is timestamps for each frame in the video
        # get the timestamps idxes for trial[0] to trial[1]
        ts_trial = ts[(ts > trial[0]) & (ts < trial[1])]
        start_idx = np.searchsorted(ts, trial[0])
        frame_idxes = np.arange(start_idx, start_idx + reg_frame_num)
        if abs(len(ts_trial) - reg_frame_num) > 10:
            raise ValueError(f'Number of frames in the video does not match the expected number of frames {reg_frame_num}. Bias > 10')
        trial_index_list.append(frame_idxes)
    session_frames = np.array(session_frames)
    trial_index_list = np.array(trial_index_list)
    # [trial, frame, height, width]
    return trial_index_list, url

def load_video(index, url, quiet=True):
    trial_frames = vidio.get_video_frames_preload(
        url, 
        index, 
        mask=np.s_[:, :, 0],
        quiet=quiet
    )
    return trial_frames

def load_whisker_video(index, url, mask, quiet=True):
    # load cropped video frames for the whisker pad ROI
    trial_frames = vidio.get_video_frames_preload(
        url, 
        index, 
        mask=mask,
        quiet=quiet,
        func=grayscale
    )
    # print(trial_frames[0,:,:,0])
    # print('-----------------')
    # print(trial_frames[0,:,:,1])
    return trial_frames

def grayscale(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
def get_dlc_midpoints(dlc_df, target):
    # Load dataframe
    # dlc_df = pd.read_parquet(dlc_pqt)
    # Set values to nan if likelihood is too low and calcualte midpoints
    idx = dlc_df.loc[dlc_df[f'{target}_likelihood'] < 0.9].index
    dlc_df.loc[idx, [f'{target}_x', f'{target}_y']] = np.nan
    if all(np.isnan(dlc_df[f'{target}_x'])) or all(np.isnan(dlc_df[f'{target}_y'])):
        raise ValueError(f'Failed to calculate midpoint, {target} all NaN in DLC data')
    else:
        mloc = [int(np.nanmean(dlc_df[f'{target}_x'])), int(np.nanmean(dlc_df[f'{target}_y']))]
        return mloc

def get_whisker_pad_roi(one, eid, camera):
    # load DLC data
    dlc_data = load_dlc_data(one, eid, camera)['dlc']
    nose_mid = get_dlc_midpoints(dlc_data, 'nose_tip')
    # Go through the different pupil points to see if any has not all NaN values
    try:
        pupil_mid = get_dlc_midpoints(dlc_data, 'pupil_top_r')
    except ValueError:
        try:
            pupil_mid = get_dlc_midpoints(dlc_data, 'pupil_left_r')
        except ValueError:
            try:
                pupil_mid = get_dlc_midpoints(dlc_data, 'pupil_right_r')
            except ValueError:
                try:
                    pupil_mid = get_dlc_midpoints(dlc_data, 'pupil_bottom_r')
                except ValueError:
                    pupil_mid = None
    assert nose_mid is not None, 'Nose midpoint is None'
    assert pupil_mid is not None, 'Pupil midpoint is None'
    anchor = np.mean([nose_mid, pupil_mid], axis=0)
    dist = np.sqrt(np.sum((np.array(nose_mid) - np.array(pupil_mid))**2, axis=0))
    w, h = int(dist / 2), int(dist / 3)
    x, y = int(anchor[0] - dist / 4), int(anchor[1])

    # Check if the mask has negative values (sign that the midpoint location is off)
    if any(i < 0 for i in [x, y, w, h]) is True:
        raise ValueError(f"ROI for motion energy on {camera}Camera could not be computed. "
                         f"Check for issues with the raw video or dlc output.")
    # Note that x and y are flipped when loading with cv2, therefore:
    mask = np.s_[y:y + h, x:x + w]
    roi = np.asarray([w, h, x, y])
    return roi, mask

def load_behavior(one, eid, target, idx, camera='left', collection='alf'):
    """
    Load behavior data from a specific camera for a given session.

    Parameters
    ----------
    one : 
    eid : str
    camera : str
        'leftCamera' | 'rightCamera'
    attribute : list
        e.g., ['dlc', 'features', 'times']
    collection : str, optional
        'alf' | 'raw_video_frames' | 'raw_video_data'

    Returns
    -------
    dict
        'times': timestamps for behavior signal
        'values': associated values
        'skip': bool, True if there was an error loading data
    """

    # To load wheel and motion energy, we just use the SessionLoader, e.g.
    sess_loader = SessionLoader(one, eid=eid)
    
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    avail_targets = [
        'wheel-position', 'wheel-velocity', 'wheel-speed',
        'whisker-motion-energy', 'left-pupil-diameter', 'right-pupil-diameter',
    ]
    assert target in avail_targets, f'{target} not in {avail_targets}'
    if target == 'wheel-position':
        sess_loader.load_wheel()
        return sess_loader.wheel['position'].to_numpy()[idx]
    elif target == 'wheel-velocity':
        sess_loader.load_wheel()
        return sess_loader.wheel['velocity'].to_numpy()[idx]
    elif target == 'wheel-speed':
        sess_loader.load_wheel()
        return np.abs(sess_loader.wheel['velocity'].to_numpy()[idx])
    elif target == 'whisker-motion-energy':
        sess_loader.load_motion_energy(views=[camera])
        return sess_loader.motion_energy[f'{camera}Camera']['whiskerMotionEnergy'].to_numpy()[idx]
    elif target == 'left-pupil-diameter':
        dlc_left = one.load_object(eid, "leftCamera", attribute=["dlc", "features", "times"], collection=collection)
        return dlc_left.features.pupilDiameter_smooth.to_numpy()[idx]
    elif target == 'right-pupil-diameter':
        dlc_right = one.load_object(eid, "rightCamera", attribute=["dlc", "features", "times"], collection=collection)
        return dlc_right.features.pupilDiameter_smooth.to_numpy()[idx]
    else:
        raise NotImplementedError

def get_optic_flow(video, save_path=None, fps=60, ses='', trial=''):
    vec_heatmap = []
    vec_field = []
    scale = 5  # scale for drawing arrows
    step_size = 16
    h, w = video[0].shape[:2]
    video = video.astype(np.float32)
    raw_video = video.copy()
    me = np.mean(np.abs(np.diff(video, axis=0)), axis=(1, 2))
    # normalize the motion energy
    me = (me - np.min(me)) / (np.max(me) - np.min(me))
    for i in range(len(video) - 1):
        frame1 = video[i]
        frame2 = video[i + 1]
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Draw optical flow arrows
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                pt1 = (x, y)
                pt2 = (int(x + flow[y, x, 0] * scale), int(y + flow[y, x, 1] * scale))
                cv2.arrowedLine(video[i], pt1, pt2, (255, 0, 0), 1)
        # vec_video.append(new_frame)
        vec_heatmap.append(np.abs(flow).sum(2))
        vec_field.append(flow)
    vec_field = np.abs(vec_field)
    vec_field = np.array(vec_field) # frame, height, width, 2
    vec_x_med = np.median(vec_field[..., 0], axis=(1,2))
    vec_y_med = np.median(vec_field[..., 1], axis=(1,2))
    # only take 90% of the x, y vecors
    clip_vec_field = vec_field.copy()
    clip_vec_field[..., 0] = np.clip(clip_vec_field[..., 0], np.percentile(clip_vec_field[..., 0], 10), np.percentile(clip_vec_field[..., 0], 90))
    clip_vec_field[..., 1] = np.clip(clip_vec_field[..., 1], np.percentile(clip_vec_field[..., 1], 10), np.percentile(clip_vec_field[..., 1], 90))    
    clip_vec_field = np.mean(np.abs(clip_vec_field), axis=(1,2,3))
    vec_field = np.mean(np.abs(vec_field), axis=(1,2,3))
    # normalize the vectors
    vec_field = (vec_field - np.min(vec_field)) / (np.max(vec_field) - np.min(vec_field))
    vec_x_med = (vec_x_med - np.min(vec_x_med)) / (np.max(vec_x_med) - np.min(vec_x_med))
    vec_y_med = (vec_y_med - np.min(vec_y_med)) / (np.max(vec_y_med) - np.min(vec_y_med))
    clip_vec_field = (clip_vec_field - np.min(clip_vec_field)) / (np.max(clip_vec_field) - np.min(clip_vec_field))
    me = np.append(me, me[-1])
    vec_field = np.append(vec_field, vec_field[-1])
    vec_x_med = np.append(vec_x_med, vec_x_med[-1])
    vec_y_med = np.append(vec_y_med, vec_y_med[-1])
    clip_vec_field = np.append(clip_vec_field, clip_vec_field[-1])
    if save_path:
        fig, ax = plt.subplots()
        line_me, = ax.plot([], [])
        line_of, = ax.plot([], [])
        line_clip_of, = ax.plot([], [])
        line_me.set_color('r')
        line_of.set_color('b')
        line_clip_of.set_color('g')
        # set legend
        ax.legend(['Motion Energy', 'Optical Flow', 'Clip OF[0.1, 0.9]'],loc='upper left')
        # Text Object
        frame_text = fig.text(0.02, 0.95, '', fontsize=12)
        # set title
        ax.set_title(f'{ses}-t{trial} ME and Mean OF')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(me))
        # set x ticks size
        ax.tick_params(axis='x')
        # set y ticks size
        ax.tick_params(axis='y')
        def init():
            line_me.set_data([], [])
            line_of.set_data([], [])
            line_clip_of.set_data([], [])
            return line_me, line_of
        def animate(i):
            frame = i + 1
            line_me.set_data(np.arange(frame), me[:frame])
            line_of.set_data(np.arange(frame), vec_field[:frame])
            line_clip_of.set_data(np.arange(frame), clip_vec_field[:frame])
            frame_text.set_text(f'Frame: {frame}')
            return line_me, line_of, line_clip_of
        ani = animation.FuncAnimation(fig, animate, frames=len(me), init_func=init, blit=True)
        # save as gif
        ani.save('output_ani.gif', writer='pillow', fps=fps)

        # create a new figure
        fig, ax = plt.subplots()
        line_x, = ax.plot([], [])
        line_y, = ax.plot([], [])
        line_x.set_color('r')
        line_y.set_color('b')
        # set legend
        ax.legend(['Vec X', 'Vec Y'],loc='upper left')
        # Text Object
        frame_text = fig.text(0.02, 0.95, '', fontsize=12)
        # set title
        ax.set_title(f'{ses}-t{trial} Median OF Vectors')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(vec_x_med))
        # set x ticks size
        ax.tick_params(axis='x')
        # set y ticks size
        ax.tick_params(axis='y')
        def init_vec():
            line_x.set_data([], [])
            line_y.set_data([], [])
            return line_x, line_y
        def animate_vec(i):
            frame = i + 1
            line_x.set_data(np.arange(frame), vec_x_med[:frame])
            line_y.set_data(np.arange(frame), vec_y_med[:frame])
            frame_text.set_text(f'Frame: {frame}')
            return line_x, line_y
        ani_vec = animation.FuncAnimation(fig, animate_vec, frames=len(vec_x_med), init_func=init_vec, blit=True)
        # save as gif
        ani_vec.save('output_ani_vec.gif', writer='pillow', fps=fps)
        # load the gif
        ani_gif = imageio.mimread('output_ani.gif', memtest=False)
        ani_vec_gif = imageio.mimread('output_ani_vec.gif', memtest=False)
        # convert the gif to numpy array
        ani_np = standardize_gif(ani_gif)
        ani_vec_np = standardize_gif(ani_vec_gif)
        # combine frames side by side
        h, w = ani_np[0].shape[:2]
        # resize the vid_np using cv2
        video = np.array([cv2.resize(frame, (w, h)) for frame in video])
        raw_video = np.array([cv2.resize(frame, (w, h)) for frame in raw_video])
        # make video from grayscale to rgb
        video = video_gray2rgb(video).astype(np.uint8)
        raw_video = video_gray2rgb(raw_video).astype(np.uint8)
        of_gif = np.concatenate((video, ani_np), axis=1)
        vec_gif = np.concatenate((raw_video, ani_vec_np), axis=1)
        # combine the gifs
        combined_gif = np.concatenate((vec_gif, of_gif), axis=2)
        # remove the temporary files
        os.remove('output_ani.gif')
        os.remove('output_ani_vec.gif')
        # save combined gif to mp4 format
        imageio.mimsave(save_path, combined_gif, fps=5)
    return vec_field, None

def standardize_gif(gif):
    std_frames = []
    for frame in gif:
        img = Image.fromarray(frame)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        std_frames.append(np.array(img))
    return np.array(std_frames)

def video_gray2rgb(video):
    return np.array([cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in video])
