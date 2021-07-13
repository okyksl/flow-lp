import numpy as np
import torch


class NLLLoss(torch.nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll


class LossMetric:
    """Keeps track of the loss over an epoch"""

    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        return self.running_loss / self.count

    def reset(self) -> None:
        self.running_loss = 0
        self.count = 0


class AccuracyMetric:
    """Keeps track of the top-k accuracy over an epoch

    Args:
        k (int): Value of k for top-k accuracy
    """

    def __init__(self, k: int = 1) -> None:
        self.correct = 0
        self.total = 0
        self.k = k

    def update(self, out: torch.Tensor, target: torch.Tensor) -> None:
        # Computes top-k accuracy
        _, indices = torch.topk(out, self.k, dim=-1)
        target_in_top_k = torch.eq(indices, target[:, None]).bool().any(-1)
        total_correct = torch.sum(target_in_top_k, dtype=torch.int).item()
        total_samples = target.shape[0]

        self.correct += total_correct
        self.total += total_samples

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
