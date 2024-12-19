from utils.utils import (
    get_args,
    set_seed,
    train_rrr,
    NAME2MODEL
)
from utils.dataset_utils import (
    split_dataset,
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from utils.log_utils import (
    logging
)
from utils.loss_utils import (
    loss_fn_
)
from utils.plot_utils import (
    plot_embeddings,
    plot_embeddings_anim,
    save_numpy_video_to_gif
)
from loader.make import (
    make_contrast_loader
)
from trainer.make import (
    make_contrast_trainer
)
import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import OneCycleLR
from transformers import ViTMAEConfig
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
from torch_optimizer import Lamb
import numpy as np
import cebra

def main():
    log = logging(header='(੭｡╹▿╹｡)੭', header_color='#df6da9')
    log.info('Testing!')
    # set config
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)
    # set seed
    set_seed(config.seed)
    idx = np.random.choice(119, 100, replace=False)
    sorted_idx = np.sort(idx)
    # set accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # log accelerator info
    local_rank = accelerator.state.local_process_index # rank of the process
    world_size = accelerator.state.num_processes # number of processes
    dsitributed = not accelerator.state.distributed_type.value == 'NO'
    log.info(f"Distributed: {dsitributed}, Local Rank: {local_rank}, World Size: {world_size}, Device: {accelerator.device}")
    with open(f'data/eid.txt', 'r') as f:
        eids = f.readlines()
    eids = [eid.strip() for eid in eids]
    eids = np.sort(eids).tolist()
    bps_res = []
    save_plot=args.save_plot
    for eid in eids:
        log.info(f"Processing {eid}")
        # set dataset
        transform = transforms.Compose([
            # transforms.ToTensor(),  # Convert image to tensor
            # transforms.Resize((224, 224)),  # Resize to the input size expected by the model
            transforms.Resize((144, 144)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
            transforms.Normalize(mean=[0.5], std=[0.5]) # gray scale
        ])
        dataset_split_dict = split_dataset(config.dirs.data_dir,eid=eid)

        pretrain_data_loader,_ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',
                                        eid=eid,
                                        batch_size=config.training.train_batch_size,
                                        shuffle=True,
                                        transform = transform,
                                        device = accelerator.device,
                                        idx_offset=3,
                                        mode='pretrain'
        )
        valid_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                        eid=eid,
                                        batch_size=1,
                                        shuffle=False,
                                        transform = transform,
                                        device = accelerator.device,
                                        idx_offset=3,
                                        mode='val'
        )
        train_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                        eid=eid,
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
        lr_scheduler = None

        # set criterion
        criterion = loss_fn_
        # set trainer
        trainer_kwargs = {
            "log_dir": config.dirs.log_dir,
            "accelerator": accelerator,
            "lr_scheduler": lr_scheduler,
            "config": config,
            "criterion": criterion,
            "dataset_split_dict": dataset_split_dict,
            "eid": eid,
            "max_steps": max_steps,
            "log": log,
            "use_wandb": False,
            "val_data_loader": valid_data_loader,
            "train_data_loader": train_data_loader,
        }
        trainer = make_contrast_trainer(
            model=model,
            optimizer=optimizer,
            data_loader=pretrain_data_loader,
            **trainer_kwargs
        )

        test_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                        eid=eid,
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
            fig.savefig(f'{args.model}_{eid[:5]}_embed.png')
            train_data = {
                eid:
                {
                    "X": [], 
                    "y": [], 
                    "setup": {}
                } 
            }
            train_data[eid]["X"].append(train_embedding[:, sorted_idx,:])
            train_data[eid]["X"].append(test_embedding[:, sorted_idx,:])
            train_data[eid]["y"].append(train_neural)
            train_data[eid]["y"].append(test_neural)
            log.info(f"Train Data Shape: {train_data[eid]['X'][0].shape}, Neural Data Shape: {train_data[eid]['y'][0].shape}")
            log.info(f"Test Data Shape: {train_data[eid]['X'][1].shape}, Neural Data Shape: {train_data[eid]['y'][1].shape}")
            # make mask ratio 0
            rrr_result = train_rrr(train_data)
            test_bps = np.nanmean(rrr_result[eid]['bps'])
            log.info(f"{eid} Test BPS: {test_bps}")
            bps_res.append(test_bps)
            if not save_plot:
                continue
            fig, axes = plot_embeddings(
                embeddings=test_embedding[0],
                title=f"{args.model}_{eid[:5]}_embed_test",
            )
            fig.savefig(f'test_embed_{args.model}_{eid[:5]}.png')
            test_data_loader, _ = make_contrast_loader('/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_whisker-video.h5',                               
                                            eid=eid,
                                            batch_size=1,
                                            shuffle=False,
                                            transform = None,
                                            device = accelerator.device,
                                            idx_offset=3,
                                            mode='test'
            )
            for idx, batch in enumerate(test_data_loader):
                assert batch['ref'].shape[0] == 1
                video = batch['ref'].squeeze(0).cpu().numpy()
                save_numpy_video_to_gif(video, f'test_{args.model}_{eid[:5]}_{idx}.gif',fps=10)
                # save embedding to gif
                plot_embeddings_anim(
                    embeddings=test_embedding[idx],
                    title=f"{args.model}_{eid[:5]}_embed_test_{idx}",
                    outfile=f'test_embed_{args.model}_{eid[:5]}_{idx}.gif',
                    fps=10
                )
                if idx > 3:
                    break
    for i in range(len(bps_res)):
        # print 2 digits
        print(f'{bps_res[i]:.5f}')
    print(f'{np.mean(bps_res):.5f}')
        # np.save(f'data/data_rrr_{args.model}_{args.eid[:5]}', train_data)
if __name__ == '__main__':
    main()
