import torch
from torch.utils.data import DataLoader
from loader.base import BaseDataset
from loader.contrast import ContrastDataset
from utils.dataset_utils import load_h5_file

def make_loader(config, dataset_split_dict, accelerator=None):
    train_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='train',
        accelerator=accelerator
    )
    train_dataloader = train_dataset.get_dataloader()

    val_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='val',
        accelerator=accelerator
    )
    val_dataloader = val_dataset.get_dataloader()

    test_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='test',
        accelerator=accelerator
    )
    test_dataloader = test_dataset.get_dataloader()
    return train_dataloader, val_dataloader, test_dataloader

def make_contrast_loader(dataset_path,
                         mode='pretrain',
                         eid=None,
                         batch_size=512,
                         shuffle=True,
                         transform=None,
                         device='cuda',
                         idx_offset=4,
                         time_offset=None,
                         num_workers=4):
    data = load_h5_file(dataset_path, eid)
    dataset = ContrastDataset(
        data[eid],
        mode=mode,
        transform=transform,
        idx_offset=idx_offset,
        time_offset=time_offset,
        device=device
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # pin_memory=True,
        # num_workers=1
    )

    return dataloader, 1