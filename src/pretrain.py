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
    info_nce
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
from transformers import ViTMAEConfig, AutoImageProcessor
from torchvision import transforms

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
    _, _, test_dataloader = make_loader(config, dataset_split_dict)
    meta_data = get_metadata_from_loader(test_dataloader, config)
    log.info(f"meta_data: {meta_data}")
    data_loader = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',
                                       eid=args.eid,
                                       batch_size=128,
                                       shuffle=True,
                                       transform = transform,
    )
    # set model
    modle_config = ViTMAEConfig(**config.model)
    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(modle_config)

    # set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.optimizer.lr, 
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps
    )
    # set scheduler
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        total_steps=len(dataset_split_dict['train']) // config.training.train_batch_size * config.training.num_epochs,
        max_lr=config.optimizer.lr,
        pct_start=config.optimizer.warmup_pct,
        div_factor=config.optimizer.div_factor,
    )

    # set criterion
    criterion = info_nce
    # set accelerator
    accelerator = Accelerator()
    model, optimizer, lr_scheduler= accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    # set trainer
    trainer_kwargs = {
        "log_dir": config.dirs.log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "config": config,
        "criterion": criterion,
        "dataset_split_dict": dataset_split_dict,
        "eid": args.eid,
        "max_steps": 10000,
        "log": log,
    }
    trainer = make_contrast_trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        **trainer_kwargs
    )
    trainer.fit()
if __name__ == '__main__':
    main()
