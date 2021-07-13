import warnings

import torch


class WarmupLR(torch.optim.lr_scheduler.LambdaLR):
    """Increases the learning rate linearly and then keeps it constant.
    When last_epoch=-1, sets initial lr as lr.
    Call scheduler at each optimizer step.
    Sets the learning rate to:
        lr = initial_lr * min(1.0, (step / warmup_steps))
    Args:
        optimizer (Optimizer): Wrapped optimizer
        warmup_steps (int): The number of warmup steps. 
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, last_epoch: int = -1,
    ) -> None:
        decay = lambda step: min(1.0, step / warmup_steps)
        super().__init__(optimizer, lr_lambda=decay, last_epoch=last_epoch)

    def load_state_dict(self, state_dict: dict) -> None:
        # Bypass save state warning from _LambdaLR scheduler
        with warnings.catch_warnings(record=True):
            super().load_state_dict(state_dict)

    def state_dict(self) -> dict:
        # Bypass save state warning from _LambdaLR scheduler
        # More info: https://github.com/pytorch/pytorch/issues/46405
        with warnings.catch_warnings(record=True):
            return super().state_dict()


class ExponentialDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """Implements ExponentialDecay scheduler from Keras.
    To match Keras behaviour, call scheduler at each optimizer step
    (usually at each iteration).
    When last_epoch=-1, sets initial lr as lr.
    Sets the learning rate to:
    lr = initial_lr * (decay_rate ^ (step / decay_step))
    If staircase is True, step / decay_steps is an integer division.
    Args:
        optimizer (Optimizer): Wrapped optimizer
        decay_steps (int): Must be positive, see the decay computation above.
        decay_rate (float): The decay rate
        staircase (bool): If True, the learning rate is decayed at discrete intervals.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        last_epoch: int = -1,
    ) -> None:

        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

        if staircase:
            decay = lambda step: decay_rate ** (int(step / decay_steps))
        else:
            decay = lambda step: decay_rate ** (step / decay_steps)

        super().__init__(optimizer, lr_lambda=decay, last_epoch=last_epoch)

    def load_state_dict(self, state_dict: dict) -> None:
        # Bypass save state warning from _LambdaLR scheduler
        with warnings.catch_warnings(record=True):
            super().load_state_dict(state_dict)

    def state_dict(self) -> dict:
        # Bypass save state warning from _LambdaLR scheduler
        # More info: https://github.com/pytorch/pytorch/issues/46405
        with warnings.catch_warnings(record=True):
            return super().state_dict()
