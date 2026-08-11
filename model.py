import torch
import torch.nn as nn


class CNNCifar(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=16,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=16,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu2 = nn.ReLU()

        self.maxPooling1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(
            num_features=32,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(
            num_features=32,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu4 = nn.ReLU()

        self.maxPooling2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn5 = nn.BatchNorm2d(
            num_features=64,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn6 = nn.BatchNorm2d(
            num_features=64,
            eps=1e-4,
            affine=True,
            track_running_stats=True
        )

        self.relu6 = nn.ReLU()

        self.maxPooling3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=64 * 4 * 4,
            out_features=512
        )

        self.relu7 = nn.ReLU()

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=10
        )

    def forward(self, data):

        data = self.conv1(data)
        data = self.bn1(data)
        data = self.relu1(data)

        data = self.conv2(data)
        data = self.bn2(data)
        data = self.relu2(data)

        data = self.maxPooling1(data)

        data = self.conv3(data)
        data = self.bn3(data)
        data = self.relu3(data)

        data = self.conv4(data)
        data = self.bn4(data)
        data = self.relu4(data)

        data = self.maxPooling2(data)

        data = self.conv5(data)
        data = self.bn5(data)
        data = self.relu5(data)

        data = self.conv6(data)
        data = self.bn6(data)
        data = self.relu6(data)

        data = self.maxPooling3(data)
        data = self.flatten(data)

        data = self.fc1(data)
        data = self.relu7(data)

        data = self.dropout(data)

        data = self.fc2(data)

        return data