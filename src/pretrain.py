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
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import OneCycleLR
from transformers import ViTMAEConfig
from torchvision import transforms
from torch_optimizer import Lamb
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
    # set accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # log accelerator info
    local_rank = accelerator.state.local_process_index # rank of the process
    world_size = accelerator.state.num_processes # number of processes
    dsitributed = not accelerator.state.distributed_type.value == 'NO'
    log.info(f"Distributed: {dsitributed}, Local Rank: {local_rank}, World Size: {world_size}, Device: {accelerator.device}")

    # set dataset
    transform = transforms.Compose([
        # transforms.ToTensor(),  # Convert image to tensor
        # transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.Resize((144, 144)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
        transforms.Normalize(mean=[0.5], std=[0.5]) # gray scale
    ])
    dataset_split_dict = split_dataset(config.dirs.data_dir,eid=args.eid)
    train_num = len(dataset_split_dict['train'])
    test_num = len(dataset_split_dict['test'])
    train_idx = list(range(train_num))
    test_idx = list(range(train_num, train_num + test_num))

    pretrain_data_loader,_ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',
                                       eid=args.eid,
                                       batch_size=config.training.train_batch_size,
                                       shuffle=True,
                                       transform = transform,
                                       device = accelerator.device,
                                       idx_offset=3,
                                       mode='pretrain'
    )
    valid_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                       eid=args.eid,
                                       batch_size=1,
                                       shuffle=False,
                                       transform = transform,
                                       device = accelerator.device,
                                       idx_offset=3,
                                       mode='val'
    )
    train_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                       eid=args.eid,
                                       batch_size=1,
                                       shuffle=False,
                                       transform = transform,
                                       device = accelerator.device,
                                       idx_offset=3,
                                       mode='train'
    )
    # set model
    model_name = args.model
    if model_name == 'c':
        model_name = 'ContrastViT'
    elif model_name == 'cm':
        model_name = 'ContrastViTMAE'
    elif model_name == 'm':
        model_name='MAE'
    model_class = NAME2MODEL[model_name]
    model = model_class(config.model)

    # set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.optimizer.lr, 
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps
    )

    # set scheduler
    max_steps = 40000
    global_batch_size = config.training.train_batch_size * world_size
    max_lr = config.optimizer.lr * accelerator.num_processes
    num_epochs = max_steps // len(pretrain_data_loader)
    log.info(f"Max Steps: {max_steps}, Num Epochs: {num_epochs}, Max LR: {max_lr}, Global Batch Size: {global_batch_size}, World Size: {world_size}")
    # lr_scheduler = OneCycleLR(
    #     optimizer=optimizer,
    #     total_steps=max_steps,
    #     max_lr=max_lr,
    #     pct_start=2 / num_epochs,
    #     final_div_factor=1000,
    # )
    lr_scheduler = None

    # set criterion
    # temperature is to control the sharpness of the distribution
    # the smaller the temperature, the sharper the distribution
    # criterion = loss_fn(temperature=0.1)
    criterion = loss_fn_
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
        "val_data_loader": valid_data_loader,
        "train_data_loader": train_data_loader,
    }
    trainer = make_contrast_trainer(
        model=model,
        optimizer=optimizer,
        data_loader=pretrain_data_loader,
        **trainer_kwargs
    )
    trainer.fit()
    # get test_data_loader
    test_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                       eid=args.eid,
                                       batch_size=1,
                                       shuffle=False,
                                       transform = transform,
                                       device = accelerator.device,
                                       idx_offset=3,
                                       mode='test'
    )
    # get embedding
    if accelerator.is_main_process:
        train_embedding, train_neural = trainer.transform(
            train_data_loader, 
            return_neural=True,
            use_best=True
            )
        test_embedding, test_neural = trainer.transform(
            test_data_loader, 
            return_neural=True,
            use_best=True
            )
        train_embedding, train_neural = train_embedding.cpu().numpy(), train_neural.cpu().numpy()
        test_embedding, test_neural = test_embedding.cpu().numpy(), test_neural.cpu().numpy()
        train_n, val_n = train_neural.shape[0], test_neural.shape[0]
        e_dim = train_embedding.shape[-1]
        # reshape the embeddings
        train_embedding = train_embedding.reshape((train_n, -1, e_dim))
        test_embedding = test_embedding.reshape((val_n, -1, e_dim))
        log.info(f"Transformed Train Embedding Shape: {train_embedding.shape}, Train Neural Shape: {train_neural.shape}, Test Embedding Shape: {test_embedding.shape}, Test Neural Shape: {test_neural.shape}")
        ax = cebra.plot_embedding(train_embedding)
        fig = ax.get_figure()
        fig.savefig(f'{args.model}_{args.eid[:5]}_embed.png')
        train_data = {
            args.eid:
            {
                "X": [], 
                "y": [], 
                "setup": {}
            } 
        }
        train_data[args.eid]["X"].append(train_embedding)
        train_data[args.eid]["X"].append(test_embedding)
        train_data[args.eid]["y"].append(train_neural)
        train_data[args.eid]["y"].append(test_neural)
        log.info(f"Train Data Shape: {train_data[args.eid]['X'][0].shape}, Neural Data Shape: {train_data[args.eid]['y'][0].shape}")
        log.info(f"Test Data Shape: {train_data[args.eid]['X'][1].shape}, Neural Data Shape: {train_data[args.eid]['y'][1].shape}")
        np.save(f'data/data_rrr_{args.model}_{args.eid[:5]}', train_data)
if __name__ == '__main__':
    main()
