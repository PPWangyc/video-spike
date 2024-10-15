import os
import random

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    """
    # Get the list of filenames
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.tar')]

    # Shuffle the filenames
    random.shuffle(filenames)

    # Split the data into train, val, and test sets
    split1 = int(train_ratio * len(filenames))
    split2 = int((train_ratio + val_ratio) * len(filenames))
    train_filenames = filenames[:split1]
    val_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]

    return {
        'train': train_filenames,
        'val': val_filenames,
        'test': test_filenames
    }