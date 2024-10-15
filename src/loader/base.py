import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import webdataset as wds
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

class BaseDataset():
    def __init__(
            self, 
            config,
            data, 
            mode='train',
            ):
        self.config = config
        self.mode = mode
        dataset = wds.WebDataset(data[mode])
        if mode == 'train':
            dataset = dataset.shuffle(1000)
        self.dataset = dataset.decode("pil").map(self.preprocess_sample)
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.data.modalities.video.height, config.data.modalities.video.width))
            ])
    
    def preprocess_sample(
            self,
            sample
            ):
        out = dict(__key__=sample["__key__"])

        for key, value in sample.items():
            k = key.split('.pyd')[0]
            if k in self.config.data.modalities:
                out[k] = self.process_modalities(value=value, mod=k)
        return out
    
    def process_modalities(
            self,
            value,
            mod
            ):
        if mod == 'ap':
            return torch.from_numpy(value)
        elif mod == 'video':
            if self.config.data.modalities[mod]:
                value= torch.stack([self.video_transform(v) for v in value])
                # save tensor to image
                # save_image(value[0], 'test.png')
                return value
                

    def get_dataloader(
            self,
            ):
        return DataLoader(
            self.dataset,
            batch_size=self.config.training.train_batch_size if self.mode == 'train' else self.config.training.test_batch_size,
            num_workers=self.config.training.num_workers,
            )