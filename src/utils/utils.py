import argparse
import os
import torch
import numpy as np
import random
from utils.config_utils import ParseKwargs, config_from_kwargs


def get_args():
    parser = argparse.ArgumentParser(description='IBL Spike Video Project')
    parser.add_argument('--model_config', type=str, default='configs/model/model_config.yaml', help='Model config file')
    parser.add_argument('--train_config', type=str, default='configs/train/train_config.yaml', help='Train config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    return args

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))