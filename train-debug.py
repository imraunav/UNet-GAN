"""
This script is for debugging and finding where does the nan values appear
- Train a discriminator to differentiate noise and x-ray image
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
import numpy as np


from models.unet import UNet
# from models.unet_sp import UNet_SP

from models.conv_discriminator import Discriminator
from utils import XRayDataset

import hyperparameters

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(100352, 512)  # Assuming input size is 86x86
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # any unused port

    # initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_loader(world_size):
    dataset = XRayDataset(
        hyperparameters.dataset_path
    )  # define the class and then use function
    sampler = DistributedSampler(dataset, num_replicas=world_size, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=hyperparameters.num_workers,
        sampler=sampler,
    )
    return loader, sampler


def main(rank, world_size):
    print(f"Running training on GPU {rank}")
    ddp_setup(rank, world_size)

    # generator = UNet(n_channels=2, n_classes=1).to(rank)
    # generator = DDP(generator, device_ids=[rank])

    # discriminator = UNet_SP(n_channels=1, n_classes=1).to(rank)
    discriminator = CNNClassifier(1).to(rank)
    discriminator = DDP(discriminator, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, discriminator, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        gpu_id,
        discriminator,
        dataloader,
        datasampler,
    ) -> None:
        print(f"Initializing trainer on GPU {gpu_id}")
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.adv_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.optimizer = ZeroRedundancyOptimizer(
            self.discriminator.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )

    def _save_checkpoint(self, epoch: int, d_loss: int):
        print(f"Checkpoint reached ar epoch {epoch}!")

        # if not os.path.exists("./weights"):
        #     os.mkdir("./weights")

        # ckp = self.discriminator.module.state_dict()
        # model_path = f"./weights/discriminator_{epoch}_loss{d_loss:.4f}.pt"
        # torch.save(ckp, model_path)

    def _on_epoch(self, epoch: int):
        self.datasampler.set_epoch(epoch)
        epoch_loss_d = []
        for batch in self.dataloader:
            batch_loss_d = self._on_batch(batch)
            epoch_loss_d.append(batch_loss_d)

        return np.mean(epoch_loss_d)

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)
        if hyperparameters.debug:
            print("Low energy image range: ", low_imgs.min(), low_imgs.max())
        batch_loss_d = self._train_discriminator(low_imgs, high_imgs)
        return batch_loss_d

    def _train_discriminator(self, low_imgs, high_imgs):
        losses = []
        # low_imgs, high_imgs = batch
        for imgs, label in zip([low_imgs, high_imgs], [1, 0]):
            pred_labels = self.discriminator(imgs).sigmoid()
            target_labels = torch.full(
                pred_labels.shape, label, dtype=pred_labels.dtype, device=self.gpu_id
            )
            loss = self.adv_loss(target_labels, pred_labels)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     self.discriminator.parameters(),
            #     max_norm=10,
            #     norm_type=2.0,
            #     error_if_nonfinite=True,
            #     foreach=None,
            # )

            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def train(self, max_epoch):
        self.discriminator.train()

        for epoch in range(max_epoch):
            epoch_loss_d = self._on_epoch(epoch)
            self.loss_writer(epoch, epoch_loss_d)
            if epoch % hyperparameters.ckpt_per:
                self._save_checkpoint(epoch, epoch_loss_d)

        # Final epoch save
        self._save_checkpoint(max_epoch, epoch_loss_d)

    def loss_writer(self, epoch, epoch_loss_d):
        print("Epoch : ", epoch, "Discriminator loss : ", epoch_loss_d)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )
