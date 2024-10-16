import torch
from torch.utils.data import DataLoader
from loader.base import BaseDataset

def make_loader(config, dataset_split_dict):
    train_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='train'
    )
    train_dataloader = train_dataset.get_dataloader()

    val_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='val'
    )
    val_dataloader = val_dataset.get_dataloader()

    test_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='test'
    )
    test_dataloader = test_dataset.get_dataloader()
    return train_dataloader, val_dataloader, test_dataloader