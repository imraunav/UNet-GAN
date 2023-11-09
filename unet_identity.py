"""
This is a dummy attempt to see if unet learning is ok
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
from utils import XRayDataset

import hyperparameters


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

    unet = UNet(n_channels=2, n_classes=2).to(rank)
    unet = DDP(unet, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, unet, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        gpu_id,
        unet,
        dataloader,
        datasampler,
    ) -> None:
        print(f"Initializing trainer on GPU {gpu_id}")
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.unet = unet
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.adv_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.optimizers = ZeroRedundancyOptimizer(
            self.unet.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )


    def _on_epoch(self, epoch: int):
        self.datasampler.set_epoch(epoch)
        epoch_loss = []
        for batch in self.dataloader:
            batch_loss = self._on_batch(batch)
            epoch_loss.append(batch_loss)

        return np.mean(epoch_loss)

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)
        in_imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        with torch.autograd.detect_anomaly(check_nan=True):
            gen_imgs = self.generator(
                in_imgs
            ).detach()  # don't want to track grads for this yet
            loss = self.l1_loss(gen_imgs, in_imgs)
            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()
        return loss.item()

    def train(self, max_epoch):
        self.unet.train()

        for epoch in range(max_epoch):
            epoch_loss = self._on_epoch(epoch)
            self.loss_writer(epoch, epoch_loss)

        # Final epoch save
        # self._save_checkpoint(max_epoch, epoch_loss_d, epoch_loss_g)

    def loss_writer(self, epoch, epoch_loss):
        # with open("unetidentity_loss.txt", mode="a") as file:
            # file.write(f"Epoch{epoch}: {epoch_loss}\n")
        print((f"Epoch{epoch}: {epoch_loss}"))


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  #
