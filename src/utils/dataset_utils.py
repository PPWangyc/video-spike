import os
import random
import torch
import h5py
import numpy as np

def load_h5_file(file_path, eid=None):
    """
    Load an h5 file.
    """
    # if eid not a list, convert to list
    if type(eid) == str:
        eids = [eid]
    with h5py.File(file_path, 'r') as file:
        # List all groups
        eids = list(file.keys()) if eid is None else eids
        train_data = {
            eid : {
                "X" : [file[eid]['X_train'][()], file[eid]['X_test'][()]],
                "y" : [file[eid]['y_train'][()], file[eid]['y_test'][()]],
                "setup" : {}
            } for eid in eids
        }
        for eid in eids:
            n, t, c, h, w = train_data[eid]["X"][0].shape
            train_data[eid]["X"][0] = train_data[eid]["X"][0].reshape(n * t, c, h, w)
            n, t, c, h, w = train_data[eid]["X"][1].shape
            train_data[eid]["X"][1] = train_data[eid]["X"][1].reshape(n * t, c, h, w)
            train_data[eid]["X"] = np.concatenate(train_data[eid]["X"], axis=0)
            train_data[eid]["y"] = np.concatenate(train_data[eid]["y"], axis=0)
        return{
            eid : {
                "X" : train_data[eid]["X"],
                "y" : train_data[eid]["y"],
            } for eid in eids
        }
        

def split_dataset(data_dir, eid, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    """
    # Get the list of filenames
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.tar')]

    if type(eid) == str:
        eid = [eid]
    filenames = [f for f in filenames if any(e in f for e in eid)]
    # Filter the filenames by experiment ID
    # filenames = [f for f in filenames if eid in f]
    print(f"Found {len(filenames)} files for EID: {eid}")

    # Shuffle the filenames
    random.shuffle(filenames)

    # Split the data into train, val, and test sets
    split1 = int(train_ratio * len(filenames))
    split2 = int((train_ratio + val_ratio) * len(filenames))
    train_filenames = filenames[:split1]
    val_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]

    train_eids = get_eids_from_filenames(train_filenames)
    val_eids = get_eids_from_filenames(val_filenames)
    test_eids = get_eids_from_filenames(test_filenames)

    return {
        'train': train_filenames,
        'val': val_filenames,
        'test': test_filenames,
        'eid': {
            'train': train_eids,
            'val': val_eids,
            'test': test_eids
        }
    }

def get_eids_from_filenames(filenames):
    """
    Get the experiment IDs from the filenames.
    """
    eids = [os.path.basename(f).split('_')[0] for f in filenames]
    # unique eids
    eids = list(set(eids))
    return eids

def get_metadata_from_loader(data_loader, config):
    """
    Get the metadata from the data loader.
    """
    batch = next(iter(data_loader))
    input_mods = []
    for mod in config.data.modalities.keys():
        if config.data.modalities[mod]['input']:
            input_mods.append(mod)
    
    _input = []
    for mod in input_mods:
        _input.append(batch[mod].flatten(1))
    _input = torch.cat(_input, dim=-1)

    return{
        'num_neurons': batch['ap'].shape[2],
        'input_dim': _input.shape[1],
        'input_mods': input_mods,
        'output_dim': batch['ap'].shape[1] * batch['ap'].shape[2]
    }