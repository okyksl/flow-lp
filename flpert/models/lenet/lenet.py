import torch
import torch.nn as nn

from flpert.utils import truncated_normal


class LeNet(nn.Module):
    """LeNet-5 from `"Gradient-Based Learning Applied To Document Recognition"
    <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_
    """

    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self._init_weights()

    # ref: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # truncated normal distribution with std 0.1 (truncate > 2 x std)
                # https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal
                weights = truncated_normal(list(m.weight.shape), threshold=0.1 * 2)
                weights = torch.from_numpy(weights)
                m.weight.data.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.relu(self.conv1(x)))
        x = self.avgpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
