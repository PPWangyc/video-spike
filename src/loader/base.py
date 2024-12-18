import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import webdataset as wds
import webdataset.filters as filters
import numpy as np
import torchvision.transforms as transforms
from torchvision.io import write_video
import time
import random
class BaseDataset():
    def __init__(
            self, 
            config,
            data, 
            accelerator=None,
            mode='train',
            ):
        self.config = config
        self.mode = mode
        dataset = wds.WebDataset(data[mode],seed=config.seed, shardshuffle=True)
        if mode == 'train':
            dataset = dataset.shuffle(10000, rng=random.Random(config.seed))
        self.dataset = dataset.decode(wds.autodecode.torch_video, "torchrgb").map(self.preprocess_sample)
        
        self.video_transform = transforms.Compose([
            transforms.Resize((config.data.modalities.video.height, config.data.modalities.video.width))
            ])
    
    def preprocess_sample(
            self,
            sample
            ):
        out = dict(__key__=sample["__key__"])

        for key, value in sample.items():
            if any(mod in key for mod in self.config.data.modalities.keys()):
                k = key.split('.')[0]
                out[k] = self.process_modalities(value=value, mod=k).float()
        out['eid'] = sample["__key__"].split('_')[0]
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
                video, _, meta = value
                # grayscale video only take the first channel
                video = video[:,:,:,0].unsqueeze(1)
                # transform video to (T, C, H, W)
                # video= torch.stack([self.video_transform(v) for v in video]).float()

                # print(f"Time taken to process video: {end-start}")
                # _video = video.clone().squeeze(1).unsqueeze(-1)
                # # video shape: (T, C, H, W)
                # _video = _video.repeat(1,1,1,3)
                # write_video('video.mp4', _video, fps=60)
                return video
        elif mod == 'whisker-video':
            video, _, meta = value
            # grayscale video only take the first channel
            video = video[:,:,:,0].unsqueeze(1)
            return video
        elif mod == 'whisker-of':
            a = torch.from_numpy(value)
            return torch.from_numpy(value)
        elif mod == 'whisker-of-2d':
            return torch.from_numpy(value)
        elif mod == 'whisker-of-video':
            return torch.from_numpy(value)
        elif mod == 'whole-of':
            return torch.from_numpy(value)
        elif mod == 'whole-of-2d':
            return torch.from_numpy(value)
        elif mod == 'whole-of-video':
            return torch.from_numpy(value)
        elif mod == 'wheel-speed':
            return torch.from_numpy(value)
        elif mod == 'whisker-motion-energy':
            return torch.from_numpy(value)
        elif mod == 'choice':
            return torch.from_numpy(value)
        elif mod == 'block':
            return torch.from_numpy(value)
        elif mod == 'timestamp':
            return torch.from_numpy(value)
        else:
            raise NotImplementedError(f"Modality {mod} not implemented")

    def get_dataloader(
            self,
            ):
        dataloader =  DataLoader(
            self.dataset,
            batch_size=self.config.training.train_batch_size if self.mode == 'train' else self.config.training.test_batch_size,
            num_workers=self.config.training.num_workers,
            )
        return dataloader