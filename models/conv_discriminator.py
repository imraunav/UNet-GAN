import torch
from torch import nn
import torch.nn.functional as F

import hyperparameters


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(
                in_features=(
                    hyperparameters.crop_size * hyperparameters.crop_size * 128
                ),
                out_features=1,
            ),
        )

    def forward(self, x):

        return self.convblock(x)
