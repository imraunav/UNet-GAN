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
import pickle


from models.unet import UNet
from models.unet_sp import UNet_SP
# from models.discriminator import Discriminator

# from models.conv_discriminator import Discriminator
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

    g = UNet(n_channels=2, n_classes=1).to(rank)
    g = DDP(g, device_ids=[rank])

    d_l = UNet_SP(n_channels=1, n_classes=1).to(rank)
    # d_l = Discriminator().to(rank)
    d_l = DDP(d_l, device_ids=[rank])

    d_h = UNet_SP(n_channels=1, n_classes=1).to(rank)
    # d_h = Discriminator().to(rank)
    d_h = DDP(d_h, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, g, d_l, d_h, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        gpu_id,
        g,
        d_l,
        d_h,
        dataloader,
        datasampler,
    ):
        print(f"Initializing trainer on GPU {gpu_id}")
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.g = g
        self.d_l = d_l
        self.d_h = d_h
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.adv_crit = nn.BCELoss()
        self.l1_crit = nn.L1Loss()

        self.optim_g = ZeroRedundancyOptimizer(
            self.g.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )
        self.optim_dl = ZeroRedundancyOptimizer(
            self.d_l.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )
        self.optim_dh = ZeroRedundancyOptimizer(
            self.d_h.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )

    def _save_checkpoint(self, epoch: int):
        print(f"Checkpoint reached ar epoch {epoch}!")

        if not os.path.exists("./weights"):
            os.mkdir("./weights")

        ckp = self.g.module.state_dict()
        model_path = f"./weights/generator-2_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.d_l.module.state_dict()
        model_path = f"./weights/d_l-2_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.d_h.module.state_dict()
        model_path = f"./weights/d_h-2_{epoch}.pt"
        torch.save(ckp, model_path)

    def update_dl(self, real_batch, fake_batch):
        self.optim_dl.zero_grad()
        # predictions
        real_pred = torch.sigmoid(self.d_l(real_batch))

        # prep labels
        real_labels = torch.full(
            real_pred.shape,
            1,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )

        real_loss = self.adv_crit(real_pred, real_labels)
        real_loss.backward()
        self.optim_dl.step()

        self.optim_dl.zero_grad()
        fake_pred = torch.sigmoid(self.d_l(fake_batch.detach()))
        fake_labels = torch.full(
            fake_pred.shape,
            0,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )
        fake_loss = self.adv_crit(fake_pred, fake_labels)
        fake_loss.backward()
        self.optim_dl.step()

        loss = real_loss + fake_loss
        return loss.item()

    def update_dh(self, real_batch, fake_batch):
        self.optim_dh.zero_grad()
        # predictions
        real_pred = torch.sigmoid(self.d_h(real_batch))

        # prep labels
        real_labels = torch.full(
            real_pred.shape,
            1,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )

        real_loss = self.adv_crit(real_pred, real_labels)
        real_loss.backward()
        self.optim_dh.step()

        self.optim_dh.zero_grad()
        fake_pred = torch.sigmoid(self.d_h(fake_batch.detach()))
        fake_labels = torch.full(
            fake_pred.shape,
            0,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )
        fake_loss = self.adv_crit(fake_pred, fake_labels)
        fake_loss.backward()
        self.optim_dh.step()

        loss = real_loss + fake_loss
        return loss.item()

    def update_g(self, low_imgs, high_imgs):
        imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        self.optim_g.zero_grad()
        # generate
        fake_batch = torch.sigmoid(self.g(imgs))

        # classify
        fake_pred = torch.sigmoid(self.d_l(fake_batch))
        fake_labels = torch.full(
            fake_pred.shape,
            0,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )
        # loss_l = self.adv_crit(
        #     fake_pred, fake_labels
        # ) + hyperparameters.lam * self.l1_crit(fake_batch, low_imgs)
        loss_l = self.adv_crit(
            fake_pred, fake_labels
        ) + hyperparameters.alpha * self.l1_crit(fake_batch, low_imgs)
        loss_l.backward()
        self.optim_g.step()

        # generate
        fake_batch = torch.sigmoid(self.g(imgs))
        # classify
        fake_pred = torch.sigmoid(self.d_h(fake_batch))
        fake_labels = torch.full(
            fake_pred.shape,
            0,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )
        # loss_h = self.adv_crit(
        #     fake_pred, fake_labels
        # ) + hyperparameters.lam * self.l1_crit(fake_batch, high_imgs)
        loss_h = self.adv_crit(
            fake_pred, fake_labels
        ) + hyperparameters.beta * self.l1_crit(fake_batch, high_imgs)
        loss_h.backward()
        self.optim_g.step()
        return loss_h.item() + loss_l.item()

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)

        imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        # update discriminator
        fused_imgs = torch.sigmoid(self.g(imgs))

        # update dl
        dl_loss = 10_000
        for _ in range(hyperparameters.max_iter):
            dl_loss = self.update_dl(low_imgs, fused_imgs)
            if dl_loss <= hyperparameters.max_loss:
                break

        # update dh
        dh_loss = 10_000
        for _ in range(hyperparameters.max_iter):
            dh_loss = self.update_dh(high_imgs, fused_imgs)
            if dl_loss <= hyperparameters.max_loss:
                break

        # update g
        for _ in range(hyperparameters.max_iter):
            g_loss = self.update_g(low_imgs, high_imgs)

        return dl_loss, dh_loss, g_loss

    def _on_epoch(self, epoch):
        self.datasampler.set_epoch(epoch)
        dl_losses, dh_losses, g_losses = [], [], []
        for batch in self.dataloader:
            dl_loss, dh_loss, g_loss = self._on_batch(batch)
            dl_losses.append(dl_loss)
            dh_losses.append(dh_loss)
            g_losses.append(g_loss)

        return dl_losses, dh_losses, g_losses

    def train(self, max_epoch):
        self.d_l.train()
        self.d_h.train()
        self.g.train()

        dl_losses, dh_losses, g_losses = [], [], []
        for epoch in range(max_epoch):
            epoch_loss_dl, epoch_loss_dh, epoch_loss_g = self._on_epoch(epoch)
            dl_losses.extend(epoch_loss_dl)
            dh_losses.extend(epoch_loss_dh)
            g_losses.extend(epoch_loss_g)
            if epoch < 50:
                print(
                    f"[GPU{self.gpu_id}] Epoch:{epoch} dl_losses:{dl_losses[-1]}, dh_losses:{dh_losses[-1]}, g_losses:{g_losses[-1]}"
                )
            if epoch % hyperparameters.ckpt_per == 0 and self.gpu_id == 0:
                with open("dl_losses-2.pkl", mode="wb") as file:
                    pickle.dump(dl_losses, file, pickle.HIGHEST_PROTOCOL)
                with open("dh_losses-2.pkl", mode="wb") as file:
                    pickle.dump(dh_losses, file, pickle.HIGHEST_PROTOCOL)
                with open("g_losses-2.pkl", mode="wb") as file:
                    pickle.dump(g_losses, file, pickle.HIGHEST_PROTOCOL)
                self._save_checkpoint(epoch)

        # Final epoch save
        if self.gpu_id == 0:
            with open("dl_losses-2.pkl", mode="wb") as file:
                pickle.dump(dl_losses, file, pickle.HIGHEST_PROTOCOL)
            with open("dh_losses-2.pkl", mode="wb") as file:
                pickle.dump(dh_losses, file, pickle.HIGHEST_PROTOCOL)
            with open("g_losses-2.pkl", mode="wb") as file:
                pickle.dump(g_losses, file, pickle.HIGHEST_PROTOCOL)
            self._save_checkpoint(max_epoch)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  #
