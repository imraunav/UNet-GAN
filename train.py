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

    generator = UNet(n_channels=2, n_classes=1).to(rank)
    generator = DDP(generator, device_ids=[rank])

    # discriminator = UNet_SP(n_channels=1, n_classes=1).to(rank)
    discriminator = Discriminator().to(rank)
    discriminator = DDP(discriminator, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, generator, discriminator, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        gpu_id,
        generator,
        discriminator,
        dataloader,
        datasampler,
    ) -> None:
        print(f"Initializing trainer on GPU {gpu_id}")
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.adv_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.optimizers = {
            "generator": ZeroRedundancyOptimizer(
                self.generator.parameters(),
                optimizer_class=optim.Adam,
                lr=hyperparameters.base_learning_rate,
            ),
            "discriminator": ZeroRedundancyOptimizer(
                self.discriminator.parameters(),
                optimizer_class=optim.Adam,
                lr=hyperparameters.base_learning_rate,
            ),
        }

    def _save_checkpoint(self, epoch: int, d_loss: int, g_loss: int):
        print(f"Checkpoint reached ar epoch {epoch}!")

        if not os.path.exists("./weights"):
            os.mkdir("./weights")

        ckp = self.generator.module.state_dict()
        model_path = f"./weights/generator_{epoch}_loss{g_loss:.4f}.pt"
        torch.save(ckp, model_path)

        ckp = self.discriminator.module.state_dict()
        model_path = f"./weights/discriminator_{epoch}_loss{d_loss:.4f}.pt"
        torch.save(ckp, model_path)

    def _on_epoch(self, epoch: int):
        self.datasampler.set_epoch(epoch)
        epoch_loss_d, epoch_loss_g = [], []
        for batch in self.dataloader:
            batch_loss_d, batch_loss_g = self._on_batch(batch)
            epoch_loss_d.append(batch_loss_d)
            epoch_loss_g.append(batch_loss_g)

        return np.mean(epoch_loss_d), np.mean(epoch_loss_g)

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)
        in_imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        with torch.autograd.detect_anomaly(check_nan=True):
            gen_imgs = (
                self.generator(in_imgs).sigmoid().detach()
            )  # don't want to track grads for this yet
            batch_loss_d = self._train_discriminator(low_imgs, high_imgs, gen_imgs)
            batch_loss_g = self._train_generator(low_imgs, high_imgs)

        if hyperparameters.debug:
            if np.isnan(batch_loss_d):
                print("Discriminator batch loss is nan!")
                exit()
            if torch.isnan(batch_loss_g):
                print("Generator batch loss is nan!")
                exit()
        return batch_loss_d, batch_loss_g

    def content_loss(self, low_imgs, high_imgs, gen_imgs):
        # low_imgs, high_imgs = batch
        diff = torch.abs(low_imgs - high_imgs)  # take out similar info from images
        gamma = torch.pow(low_imgs + high_imgs, 0.5) / (
            2**0.5
        )  # enhance similar info by a non-linear tranform and normalise
        info_imgs = torch.abs(diff - gamma)  # formulate the info amount
        return self.l1_loss(gen_imgs, info_imgs)

    def _train_generator(self, low_imgs, high_imgs):
        losses = []
        # low_imgs, high_imgs = batch
        in_imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        for _ in range(hyperparameters.max_iter):
            gen_img = self.generator(in_imgs).sigmoid()
            pred_labels = (
                self.discriminator(gen_img).sigmoid().detach()
            )  # don't want to track grads for the discriminator
            target_labels = torch.full(
                pred_labels.shape, 1, dtype=torch.float32, device=self.gpu_id
            )
            adv_loss = self.adv_loss(target_labels, pred_labels)
            content_loss = self.content_loss(low_imgs, high_imgs, gen_img)
            loss = adv_loss + hyperparameters.lam * content_loss
            self.optimizers["generator"].zero_grad()
            loss.backward()
            losses.append(loss.item())
            self.optimizers["generator"].step()

            if (
                np.mean(losses) <= hyperparameters.min_loss
                or np.mean(losses) >= hyperparameters.max_loss
            ):
                break
        return np.mean(losses)

    def _train_discriminator(self, low_imgs, high_imgs, gen_imgs):
        losses = []
        # low_imgs, high_imgs = batch
        for _ in range(hyperparameters.max_iter):
            for imgs, label in zip([low_imgs, high_imgs, gen_imgs], [1, 1, 0]):
                pred_labels = self.discriminator(imgs).sigmoid()
                target_labels = torch.full(
                    pred_labels.shape, label, dtype=torch.float32, device=self.gpu_id
                )
                loss = self.adv_loss(target_labels, pred_labels)

                self.optimizers["discriminator"].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=10,
                    norm_type=2.0,
                    error_if_nonfinite=True,
                    foreach=None,
                )

                self.optimizers["discriminator"].step()

                losses.append(loss.item())

            if (
                np.mean(losses) <= hyperparameters.min_loss
                or np.mean(losses) >= hyperparameters.max_loss
            ):
                break
        return np.mean(losses)

        # fake_labels = self.discriminator(gen_images)

    def train(self, max_epoch):
        self.discriminator.train()
        self.generator.train()

        for epoch in range(max_epoch):
            epoch_loss_d, epoch_loss_g = self._on_epoch(epoch)
            self.loss_writer(epoch, epoch_loss_d, epoch_loss_g)
            if epoch % hyperparameters.ckpt_per:
                self._save_checkpoint(epoch, epoch_loss_d, epoch_loss_g)

        # Final epoch save
        self._save_checkpoint(max_epoch, epoch_loss_d, epoch_loss_g)

    def loss_writer(self, epoch, epoch_loss_d, epoch_loss_g):
        with open("Discriminator_loss.txt", mode="a") as file:
            file.write(f"Epoch{epoch}: {epoch_loss_d}\n")
        with open("Generator_loss.txt", mode="a") as file:
            file.write(f"Epoch{epoch}: {epoch_loss_g}\n")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  #
