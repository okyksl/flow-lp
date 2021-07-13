from typing import Any, Callable, List, Optional, Type, Union

import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(torchvision.models.ResNet):
    """Modifies `torchvision's ResNet implementation
    <https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html>`_
    to make it suitable for CIFAR 10/100.
    Removes or replaces some down-sampling layers to increase the size of the feature
    maps, in order to make it suitable for classification tasks on datasets with smaller
    images such as CIFAR 10/100.
    This network architecture is similar to the one used in
    `"Improved Regularization of Convolutional Neural Networks with Cutout"
    <https://arxiv.org/pdf/1708.04552.pdf>`_
    (code `here <https://github.com/uoguelph-mlrg/Cutout>`_) and in the popular
    `pytorch-cifar repository <https://github.com/kuangliu/pytorch-cifar>`_.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 100,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        # CIFAR: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1_planes = 64
        self.conv1 = nn.Conv2d(
            3, self.conv1_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove maxpool layer from forward by changing it into an identity layer
        self.maxpool = nn.Identity()


def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_, modified for CIFAR-10/100 images.
    Args:
       **kwargs: Keyword arguments, notably num_classes for the number of classes
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet20(**kwargs: Any) -> ResNet:
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_, modified for CIFAR-10/100 images.
    Args:
       **kwargs: Keyword arguments, notably num_classes for the number of classes
    """
    resnet20 = ResNet(BasicBlock, [3, 3, 3, -1], **kwargs)
    resnet20.layer4 = nn.Identity()
    resnet20.fc = nn.Linear(256 * BasicBlock.expansion, kwargs.get("num_classes", 100))
    return resnet20


def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_ modified for CIFAR-10/100 images.
    Args:
       **kwargs: Keyword arguments, notably num_classes for the number of classes
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
