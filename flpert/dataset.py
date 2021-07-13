from typing import Callable, Optional

import torchvision


class SVHN(torchvision.datasets.SVHN):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(SVHN, self).__init__(
            root,
            split="train" if train else "test",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class ImageNet32(torchvision.datasets.CIFAR10):
    """`ImageNet32 <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = ""
    url = None
    filename = None
    tgz_md5 = None
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [
        ["val_data_batch_1", ""],
        ["val_data_batch_2", ""],
        ["val_data_batch_3", ""],
        ["val_data_batch_4", ""],
        ["val_data_batch_5", ""],
        ["val_data_batch_6", ""],
        ["val_data_batch_7", ""],
        ["val_data_batch_8", ""],
        ["val_data_batch_9", ""],
        ["val_data_batch_10", ""],
    ]
    meta = None

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if download == True:
            assert "Download is not supported for ImageNet-32!"

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        self.targets = [ target - 1 for target in self.targets ]

    def _check_integrity(self) -> bool:
        print("No integrity check is available for ImageNet-32!")
        return True

    def _load_meta(self) -> None:
        print("No metadata is available for ImageNet-32!")
