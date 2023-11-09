import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2), padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear()
        )