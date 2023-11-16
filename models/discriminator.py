import torch
from torch import nn
import torch.nn.functional as F

import hyperparameters


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2), nn.LeakyReLU(0.02, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
        )
        self.linear = nn.LazyLinear(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits
