from utils import XRayDataset
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


dataset = XRayDataset("./CTP_Wires_Chargers_etc")
loader = DataLoader(dataset, batch_size=1)

i = 0
for l, h in loader:
    # print(l.shape)
    diff = torch.abs(l - h)
    gamma = torch.pow(l + h, 0.5)
    plt.subplot(2, 2, 1)
    plt.imshow(l.numpy()[0][0], cmap="grey")
    plt.subplot(2, 2, 2)
    plt.imshow(h.numpy()[0][0], cmap="grey")
    plt.subplot(2, 2, 3)
    plt.imshow(torch.abs(diff-gamma).numpy()[0][0], cmap="grey")
    # plt.subplot(2, 2, 4)
    # plt.imshow(add.numpy()[0][0], cmap="grey")
    plt.show()
    i += 1
    if i > 5:
        exit()
