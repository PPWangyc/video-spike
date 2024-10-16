from utils.utils import (
    get_args,
    set_seed,
    NAME2MODEL
)
from utils.dataset_utils import (
    split_dataset
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from loader.make import (
    make_loader
)
from trainer.make import (
    make_base_trainer
)
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR

def main():
    # set config
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)
    # set seed
    set_seed(config.seed)
    # set dataset
    dataset_split_dict = split_dataset(config.dirs.data_dir)
    train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
    # set model
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
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        total_steps=len(dataset_split_dict['train']) // config.training.train_batch_size * config.training.num_epochs,
        max_lr=config.optimizer.lr,
        pct_start=config.optimizer.warmup_pct,
        div_factor=config.optimizer.div_factor,
    )
    # set criterion
    criterion = torch.nn.PoissonNLLLoss(reduction="none", log_input=True)
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
        "dataset_split_dict": dataset_split_dict
    }
    trainer = make_base_trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        **trainer_kwargs
    )
    trainer.train()
if __name__ == '__main__':
    main()
