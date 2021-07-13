# ref: https://github.com/dmizr/phuber/blob/master/phuber/dataset.py

import collections
import logging
from typing import Callable, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from flpert.dataset import SVHN, ImageNet32
from flpert.pert import Augmentation
from flpert.transform import prepare_transform
from flpert.utils import to_clean_str


def split_dataset(dataset: Dataset, split: float, seed: int) -> Tuple[Subset, Subset]:
    """Splits dataset into a train / val set based on a split value and seed

    Args:
        dataset: dataset to split
        split: The proportion of the dataset to include in the validation split,
            must be between 0 and 1.
        seed: Seed used to generate the split

    Returns:
        Subsets of the input dataset

    """
    # Verify that the dataset is Sized
    if not isinstance(dataset, collections.abc.Sized):
        raise ValueError("Dataset is not Sized!")

    if not (0 <= split <= 1):
        raise ValueError(f"Split value must be between 0 and 1. Value: {split}")

    val_length = int(len(dataset) * split)
    train_length = len(dataset) - val_length
    splits = random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits


def get_loaders(
    cfg: DictConfig,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Initializes the training, validation, test data & loaders from config
    Args:
        cfg: Hydra config
        pert: pre-initialized flow augmentation to be used in dataloaders
    Returns:
        Tuple containing the train dataloader, validation dataloader and test dataloader
    """

    name = to_clean_str(cfg.name)

    if name == "mnist":
        dataset = MNIST
    elif name == "fmnist":
        dataset = FashionMNIST
    elif name == "cifar10":
        dataset = CIFAR10
    elif name == "cifar100":
        dataset = CIFAR100
    elif name == "svhn":
        dataset = SVHN
    elif name == "imagenet32":
        dataset = ImageNet32
    elif name == "custom":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid dataset: {name}")

    root = hydra.utils.to_absolute_path(cfg.root)

    # use average of spatial dimensions for cropping
    crop_size = (cfg.dim[1] + cfg.dim[2]) // 2

    logger = logging.getLogger()
    logger.info("Crop size {} will be used if cropping is enabled".format(crop_size))

    # Train
    if cfg.train.use:
        # always apply standarization after pert
        train_transform = prepare_transform(
            size=crop_size,
            crop=cfg.train.transform.crop,
            flip=cfg.train.transform.flip,
            padding=cfg.train.transform.padding,
            affine=cfg.train.transform.affine,
        )
        train_set = dataset(
            root, train=True, transform=train_transform, download=cfg.download
        )
        if cfg.val.split is not None:
            train_set, _ = split_dataset(
                dataset=train_set,
                split=cfg.val.split,
                seed=cfg.val.seed,
            )
            logger.info(
                "Using {:.2%} of train samples as training set in training".format(
                    1 - cfg.val.split
                )
            )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

    else:
        train_loader = None

    # Validation
    if cfg.val.use:
        if cfg.val.split is not None and cfg.val.split != 0.0:
            # always apply standarization after pert
            val_transform = prepare_transform(
                size=crop_size,
                crop=cfg.val.transform.crop,
                flip=cfg.val.transform.flip,
                padding=cfg.val.transform.padding,
                affine=cfg.train.transform.affine,
            )
            val_set = dataset(
                root, train=True, transform=val_transform, download=cfg.download
            )
            _, val_set = split_dataset(
                dataset=val_set,
                split=cfg.val.split,
                seed=cfg.val.seed,
            )
            logger.info(
                "Using {:.2%} of train samples as validation set in training".format(
                    cfg.val.split
                )
            )

            val_loader = DataLoader(
                val_set,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )

        else:
            logger = logging.getLogger()
            logger.info("No validation set will be used, as no split value was given.")
            val_loader = None
    else:
        val_loader = None

    # Test
    if cfg.test.use:
        # always apply standarization after pert
        test_transform = prepare_transform(
            size=crop_size,
            crop=cfg.test.transform.crop,
            flip=cfg.test.transform.flip,
            padding=cfg.test.transform.padding,
            affine=cfg.train.transform.affine,
        )
        test_set = dataset(
            root, train=False, transform=test_transform, download=cfg.download
        )
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.batch_size,
            shuffle=cfg.test.shuffle,
            num_workers=cfg.num_workers,
        )
    else:
        test_loader = None

    return (train_loader, val_loader, test_loader)
