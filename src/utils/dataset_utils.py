import os
import random

def split_dataset(data_dir, eid, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    """
    # Get the list of filenames
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.tar')]

    # Filter the filenames by experiment ID
    filenames = [f for f in filenames if eid in f]
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