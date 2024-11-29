import torch
from torch.utils.data import DataLoader
from loader.base import BaseDataset
from loader.contrast import ContrastDataset
from utils.dataset_utils import load_h5_file

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

def make_contrast_loader(dataset_path,eid=None,batch_size=512,shuffle=True,transform=None):
    data = load_h5_file(dataset_path, eid)
    video = data[eid]['X']
    neural_data = data[eid]['y']
    dataset = ContrastDataset(
        video,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader, neural_data