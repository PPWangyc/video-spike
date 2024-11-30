from utils.utils import (
    get_args,
    set_seed,
    NAME2MODEL
)
from utils.dataset_utils import (
    split_dataset,
    get_metadata_from_loader
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from utils.log_utils import (
    logging
)
from utils.loss_utils import (
    loss_fn,
    loss_fn_
)
from loader.make import (
    make_loader,
    make_contrast_loader
)
from trainer.make import (
    make_base_trainer,
    make_contrast_trainer
)
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from transformers import ViTMAEConfig
from torchvision import transforms
import numpy as np
import cebra

def main():
    log = logging(header='(੭｡╹▿╹｡)੭', header_color='#df6da9')
    log.info('Pretraining!')
    # set config
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)
    # set seed
    set_seed(config.seed)
    # set dataset
    transform = transforms.Compose([
        # transforms.ToTensor(),  # Convert image to tensor
        # transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.Resize((144, 144)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
        transforms.Normalize(mean=[0.485], std=[0.229]) 
    ])
    dataset_split_dict = split_dataset(config.dirs.data_dir,eid=args.eid)
    train_num = len(dataset_split_dict['train'])
    test_num = len(dataset_split_dict['test'])
    train_idx = list(range(train_num))
    test_idx = list(range(train_num, train_num + test_num))

    _, _, test_dataloader = make_loader(config, dataset_split_dict)
    meta_data = get_metadata_from_loader(test_dataloader, config)
    log.info(f"meta_data: {meta_data}")
    data_loader,_ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',
                                       eid=args.eid,
                                       batch_size=128,
                                       shuffle=True,
                                       transform = transform,
    )
    # set model
    # modle_config = ViTMAEConfig(**config.model)
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model)

    # set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.optimizer.lr, 
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps
    )
    # set scheduler
    max_steps = 10000
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        total_steps=max_steps,
        max_lr=config.optimizer.lr,
        pct_start=config.optimizer.warmup_pct,
        div_factor=config.optimizer.div_factor,
    )

    # set criterion
    # temperature is to control the sharpness of the distribution
    # the smaller the temperature, the sharper the distribution
    # criterion = loss_fn(temperature=0.1)
    criterion = loss_fn_
    # set accelerator
    accelerator = Accelerator()
    # set trainer
    trainer_kwargs = {
        "log_dir": config.dirs.log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "config": config,
        "criterion": criterion,
        "dataset_split_dict": dataset_split_dict,
        "eid": args.eid,
        "max_steps": max_steps,
        "log": log,
        "use_wandb": config.wandb.use,
    }
    trainer = make_contrast_trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        **trainer_kwargs
    )
    trainer.fit()
    data_loader, neural_data = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',
                                       eid=args.eid,
                                       batch_size=128,
                                       shuffle=False,
                                       transform = transform,
    )
    embedding = trainer.transform(data_loader).cpu().numpy()
    embedding = embedding.reshape((train_num + test_num), 120, -1)
    ax = cebra.plot_embedding(embedding)
    fig = ax.get_figure()
    fig.savefig(f'vit_{args.eid[:5]}_embed.png')
    train_X = embedding[train_idx]
    test_X = embedding[test_idx]
    train_y = neural_data[train_idx]
    test_y = neural_data[test_idx]
    train_data = {
        args.eid:
        {
            "X": [], 
            "y": [], 
            "setup": {}
        } 
    }
    train_data[args.eid]["X"].append(train_X)
    train_data[args.eid]["X"].append(test_X)
    train_data[args.eid]["y"].append(train_y)
    train_data[args.eid]["y"].append(train_y)
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
    np.save(f'data/data_rrr_vit_{args.eid[:5]}', train_data)
if __name__ == '__main__':
    main()
