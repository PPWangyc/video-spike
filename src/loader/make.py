import torch
from torch.utils.data import DataLoader
from loader.base import BaseDataset

def make_loader(config, dataset_split_dict):
    train_dataset = BaseDataset(
        config, 
        dataset_split_dict,
        mode='val'
    )
    train_dataloader = train_dataset.get_dataloader()
    return train_dataloader