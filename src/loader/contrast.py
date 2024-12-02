import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import webdataset as wds
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTMAEConfig, AutoImageProcessor
class ContrastDataset(Dataset):
    def __init__(
            self, 
            video, 
            timestamp=None,
            image_size=144,
            idx_offset=10,
            time_offset=None, # by second
            transform=None
            ):
        self.video = torch.tensor(video) / 255.0 # Normalize the video
        self.timestamp = timestamp
        if timestamp is None:
            self.timestamp = np.linspace(0, len(video)-1, len(video))
        self.idx_offset = idx_offset
        self.time_offset = time_offset

        self.transform = transform
    
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        ref_frame = self.video[idx]
        pos_frame = self.video[self._select_pos_idx(idx)]
        neg_frame = self.video[self._select_neg_idx(idx)]
        if self.transform:
            ref_frame = self.transform(ref_frame)
            pos_frame = self.transform(pos_frame)
            neg_frame = self.transform(neg_frame)
        # torch vision save the image as (C, H, W) format
        # import torchvision
        # # save ref_frame
        # torchvision.utils.save_image(ref_frame, f"ref_frame_{idx}.png")
        return {
            "ref": ref_frame,
            "pos": pos_frame,
            "neg": neg_frame
        }
    
    def _select_pos_idx(self, idx):
        """Selects a positive index based on idx_offset or time_offset."""
        if self.time_offset is None:
            # Select one of the closest 10 idx's idx as the pos_idx if time_offset is None
            start_idx = max(0, idx - self.idx_offset)
            end_idx = min(len(self.video), idx + self.idx_offset + 1)
            pos_idx = np.random.choice(range(start_idx, end_idx))
        else:
            # Select the closest idx based on time_offset
            valid_indices = np.where(abs(self.timestamp - self.timestamp[idx]) <= self.time_offset)[0]
            pos_idx = np.random.choice(valid_indices) if valid_indices.size > 0 else idx
        return pos_idx
    
    def _select_neg_idx(self, idx):
        """Selects a negative index randomly."""
        neg_idx = np.random.choice([i for i in range(len(self.video)) if i != idx])
        return neg_idx
