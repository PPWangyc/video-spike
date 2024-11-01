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
    dataset_split_dict = split_dataset(config.dirs.data_dir,eid=args.eid)
    train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
    meta_data = get_metadata_from_loader(train_dataloader, config)
    print(f"meta_data: {meta_data}")
    # set model
    model_class = NAME2MODEL[config.model.model_class]
    config['model']['encoder']['input_dim'] = meta_data['input_dim']
    config['model']['decoder']['output_dim'] = meta_data['output_dim']
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
        "dataset_split_dict": dataset_split_dict,
        "eid": args.eid,
    }
    trainer = make_base_trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        **trainer_kwargs
    )
    # import time
    # now = time.time()
    # count = 0
    # for batch in train_dataloader:
    #     print(f"batch: {count}, time: {time.time() - now}s, size: {batch['ap'].shape[0]}")
    #     now = time.time()
    # exit()
    trainer.train()
if __name__ == '__main__':
    main()
