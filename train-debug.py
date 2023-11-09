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
    discriminator = Discriminator().to(rank)
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
        low_imgs = torch.full(low_imgs.shape, 1, dtype=torch.float32, device=self.gpu_id)
        gen_imgs = torch.full(low_imgs.shape, 0, dtype=torch.float32, device=self.gpu_id)
        batch_loss_d = self._train_discriminator(low_imgs, high_imgs, gen_imgs)
        return batch_loss_d

    def _train_discriminator(self, low_imgs, high_imgs, gen_imgs):
        losses = []
        # low_imgs, high_imgs = batch
        for _ in range(hyperparameters.max_iter):
            for imgs, label in zip([low_imgs, gen_imgs], [1, 0]):
                pred_labels = self.discriminator(imgs).sigmoid()
                if hyperparameters.debug:
                    print(
                        "Discriminator in range: ", imgs.min().item(), imgs.max().item()
                    )
                    print(
                        "Discriminator pred range: ",
                        pred_labels.min().item(),
                        pred_labels.max().item(),
                    )
                target_labels = torch.full(
                    pred_labels.shape, label, dtype=torch.float32, device=self.gpu_id
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

            if (
                np.mean(losses) <= hyperparameters.min_loss
                or np.mean(losses) >= hyperparameters.max_loss
            ):
                break
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
