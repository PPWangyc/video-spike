
from trainer.base import BaseTrainer
from trainer.contrast import ContrastTrainer

def make_base_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    optimizer,
    **kwargs
):
    return BaseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        **kwargs
    )

def make_contrast_trainer(
    model,
    data_loader,
    optimizer,
    **kwargs
):
    return ContrastTrainer(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        **kwargs
    )