"""定义 benchmark 测试使用的网络、数据集、优化器、训练流程等资源."""
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, device, nn, optim
from torch.utils.data import DataLoader

from alphafed.dataset.mnist import FedMNIST

_LR = 0.01
_MOMENTUM = 0.9
_BATCH_SIZE = 128


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_conv_net() -> ConvNet:
    return ConvNet()


def get_optimizer(model: ConvNet) -> optim.Optimizer:
    return optim.SGD(model.parameters(), lr=_LR, momentum=_MOMENTUM)


def get_loss_fn() -> Callable:
    return F.cross_entropy


def get_train_dataloader(data_dir: str, client_ids: List[int]) -> DataLoader:
    return DataLoader(
        FedMNIST(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]),
            client_ids=client_ids
        ),
        batch_size=_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )


def get_test_dataloader(data_dir: str, client_ids: List[int]) -> DataLoader:
    return DataLoader(
        FedMNIST(
            data_dir,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]),
            client_ids=client_ids
        ),
        batch_size=_BATCH_SIZE,
        shuffle=False
    )


def train_an_epoch_process(model: ConvNet,
                           train_loader: DataLoader,
                           device: device,
                           optimizer: optim.Optimizer,
                           loss_fn: Callable) -> None:
    model.train()
    for data, labels in train_loader:
        data: Tensor
        labels: Tensor
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss: Tensor = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test_process(model: ConvNet,
                 test_loader: DataLoader,
                 device: device,
                 loss_fn: Callable) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0

    for data, labels in test_loader:
        data: Tensor
        labels: Tensor
        data, labels = data.to(device), labels.to(device)
        output: Tensor = model(data)
        test_loss += loss_fn(output, labels, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return avg_loss, accuracy
