import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import webdataset as wds
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTMAEConfig, AutoImageProcessor
import time
class ContrastDataset(Dataset):
    def __init__(
            self, 
            data_dict, 
            mode,
            timestamp=None,
            image_size=144,
            idx_offset=10,
            time_offset=None, # by second
            transform=None,
            device="cpu"
            ):
        """
        Initializes the dataset.

        Parameters:
        - data_dict: Dictionary containing different splits of data.
        - mode: One of ['pretrain', 'train', 'val', 'test'].
        - image_size: Size of the images (assumed square).
        - idx_offset: Frame selection offset for positive examples.
        - time_offset: Time offset for selecting positive examples, in seconds.
        - transform: Transformations to apply to frames.
        - device: The device to load data onto.
        """
        assert mode in ['pretrain', 'train', 'val', 'test'], f"Invalid mode: {mode}"
        if mode == 'pretrain':
            # Concatenate train, val, and test datasets
            video = np.concatenate([data_dict['train_X'], data_dict['val_X'], data_dict['test_X']], axis=0)
            n, t, c, h, w = video.shape
            video = video.reshape(n*t, c, h, w)
            timestamp = np.concatenate([data_dict['train_timestamp'], data_dict['val_timestamp'], data_dict['test_timestamp']], axis=0)
            timestamp = timestamp.reshape(-1)
            # Sort by timestamp
            sort_indices = np.argsort(timestamp)
            video = video[sort_indices]
            self.timestamp = timestamp[sort_indices]
        else:
            video = data_dict[f'{mode}_X']
            self.labels = data_dict[f'{mode}_y']  # Ensure labels are available for train, val, test
            self.timestamp = data_dict[f'{mode}_timestamp']

        # Convert video to tensor and normalize
        self.video = torch.tensor(video, dtype=torch.float32).div_(255.0)
        self.num_frames = self.video.shape[0]
        if timestamp is None:
            self.timestamp = np.linspace(0, len(video)-1, len(video))
        self.idx_offset = idx_offset
        self.time_offset = time_offset

        self.transform = transform
        self.mode = mode
        self.device = device
    
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        if self.mode == 'pretrain':
            ref_frame = self.video[idx].to(self.device)
            # return ref_frame
            pos_frame = self.video[self._select_pos_idx(idx)].to(self.device)
            neg_frame = self.video[self._select_neg_idx(idx)].to(self.device)
            if self.transform:
                ref_frame = self.transform(ref_frame)
                pos_frame = self.transform(pos_frame)
                neg_frame = self.transform(neg_frame)
            # torch vision save the image as (C, H, W) format
            # import torchvision
            # # save ref_frame
            # torchvision.utils.save_image(ref_frame, f"ref_frame_{idx}.png")
            # print(ref_frame.shape, pos_frame.shape, neg_frame.shape)
            return {
                "ref": ref_frame,
                "pos": pos_frame,
                "neg": neg_frame
            }
        else:
            ref_frame = self.video[idx].to(self.device)
            neural = self.labels[idx]
            if self.transform:
                ref_frame = self.transform(ref_frame)
            return {
                "ref": ref_frame,
                "neural": neural
            }
    
    def _select_pos_idx(self, idx):
        """Selects a positive index based on idx_offset or time_offset."""
        if self.time_offset is None:
            # Select one of the closest 10 idx's idx as the pos_idx if time_offset is None
            start_idx = max(0, idx - self.idx_offset)
            end_idx = min(len(self.video), idx + self.idx_offset + 1)
            pos_idx = np.random.uniform(start_idx, end_idx)
            pos_idx = int(pos_idx)
        else:
            # Select the closest idx based on time_offset
            valid_indices = np.where(abs(self.timestamp - self.timestamp[idx]) <= self.time_offset)[0]
            pos_idx = np.random.choice(valid_indices) if valid_indices.size > 0 else idx
        return pos_idx
    
    def _select_neg_idx(self, idx):
        """Selects a negative index randomly."""
        while True:
            neg_idx = np.random.uniform(0, self.num_frames)
            neg_idx = int(neg_idx)
            if neg_idx != idx:
                return neg_idx
