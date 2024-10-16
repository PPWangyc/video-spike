
from trainer.base import BaseTrainer

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
