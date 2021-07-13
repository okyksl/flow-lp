from typing import Any, Callable, List

import torch


class StandardizationLayer(torch.nn.Module):
    """Custom Standardization layer"""

    def __init__(self, mean: List[float], std: List[float], device: torch.device):
        super().__init__()
        channels = len(mean)
        self.mean = torch.tensor(mean).reshape(1, channels, 1, 1).to(device)
        self.std = torch.tensor(std).reshape(1, channels, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std


class InvertibleReshapeLayer(torch.nn.Module):
    """Reshapes the input batch-wise"""

    def __init__(self, forward_shape: List[int], backward_shape: List[int]):
        super().__init__()
        self.forward_shape = forward_shape
        self.backward_shape = backward_shape

    def forward(self, x: torch.Tensor, reverse=False):
        if reverse:
            return torch.reshape(x, self.backward_shape)
        else:
            return torch.reshape(x, self.forward_shape)


class InvertibleSequential(torch.nn.Module):
    def __init__(self, *args: Any):
        super().__init__()
        self.module = torch.nn.Sequential(*args)

        assert isinstance(
            self.module[0], InvertibleReshapeLayer
        ), "InvertibleSequential expects invertible reshape layer as the first module!"
        assert (
            len(self.module) == 2
        ), "InvertibleSequential expects invertible layer + invertible module!"

        self.reshape = self.module[0]
        self.net = self.module[1]

    def forward(self, input, reverse=False):
        input = self.reshape(input, reverse=False)
        input, sldj = self.net(input, reverse=reverse)
        input = self.reshape(input, reverse=True)
        return input, sldj
