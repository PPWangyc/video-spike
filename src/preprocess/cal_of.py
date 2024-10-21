import os
import numpy as np
from utils.dataset_utils import split_dataset
from loader.make import make_loader
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed
import cv2
import imageio

set_seed(42)

dataset_split_dict = split_dataset('../../data/ibl-video')
kwargs = {"model": "include:../../config/model/linear_video.yaml"}
config = config_from_kwargs(kwargs)
config = update_config('../../config/train/linear_video.yaml', config)
config['training']['train_batch_size'] = 1
config['training']['test_batch_size'] = 1

train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
count = 0
for batch in test_dataloader:
    vec_field = []
    video = batch['video'][0].squeeze(1)
    for i in range(video.shape[0]-1):
        prvs = video[i].numpy().astype(np.uint8)
        next = video[i+1].numpy().astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vec_field += [np.abs(flow).sum(2)]
    vec_field = np.array(vec_field)
    # normalize cv2.normalize
    vec_field = cv2.normalize(vec_field, None, 0, 255, cv2.NORM_MINMAX)
    vec_field = vec_field.astype(np.uint8)

    # left raw video, right optical flow
    video = video.numpy().astype(np.uint8)[1:]
    video = np.concatenate([video, vec_field], axis=2)
    # save video to gif
    imageio.mimsave(f'output_{count}.gif', video, fps=60, loop=0)
    
    # save video to mp4
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (video.shape[2], video.shape[1]), isColor=False)
    # for i in range(video.shape[0]):
    #     out.write(video[i])
    # out.release()
    count += 1