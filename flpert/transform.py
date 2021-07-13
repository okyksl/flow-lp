from abc import ABC, abstractmethod
from typing import Callable

import torch
import torchvision.transforms as transforms


def prepare_transform(
    size: int = 32,
    crop: bool = False,
    flip: bool = False,
    padding: int = 4,
    affine: float = 0.1,
) -> Callable:
    """PIL Image to Tensor transform with data augmentation
    Args:
        size: dataset crop size,
        crop: if True, adds random cropping
        flip: if True, adds random horizontal flip
    Returns:
        composed transform function
    """
    ts = []
    # data augmentations
    if crop:
        ts.append(transforms.RandomCrop(size=size, padding=padding))
    if flip:
        ts.append(transforms.RandomHorizontalFlip(p=0.5))
    if affine > 0:
        ts.append(transforms.RandomAffine(0, translate=(affine, affine)))

    # convert to tensor
    ts.append(transforms.ToTensor())

    return transforms.Compose(ts)
