import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from omegaconf import DictConfig, OmegaConf
from scipy.stats import truncnorm


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.

    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.

    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def truncated_normal(size, threshold=1):
    """Samples values from truncated normal distribution centered at 0
    Args:
        size: shape or amount of samples
        threshold: cut-off value for distribution
    Returns:
        numpy array of given size
    """
    return truncnorm.rvs(-threshold, threshold, size=size)


def bits_per_dims(nll, dim):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(dim)
    bpd = nll / (np.log(2) * dim)
    return bpd


def to_clean_str(s: str) -> str:
    """Keeps only alphanumeric characters and lowers them
    Args:
        s: a string
    Returns:
        cleaned string
    """
    return re.sub("[^a-zA-Z0-9]", "", s).lower()


def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")
