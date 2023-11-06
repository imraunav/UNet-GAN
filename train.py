import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.unet import UNet

import hyperparameters

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # any unused port

    # initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_loader(world_size):
    raise NotImplementedError
    dataset = Dataset  # define the class and then use function
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

    discriminator_low = UNet(n_channels=1, n_classes=1).to(rank)
    discriminator_low = DDP(discriminator_low, device_ids=[rank])

    discriminator_high = UNet(n_channels=1, n_classes=1).to(rank)
    discriminator_high = DDP(discriminator_high, device_ids=[rank])

    dataloader, datasampler = get_loader()
    trainer = Trainer(
        rank, generator, discriminator_low, discriminator_high, dataloader, datasampler
    )


class Trainer:
    def __init__(
        gpu_id,
        generator,
        discriminator_low,
        discriminator_high,
        dataloader,
        datasampler,
    ) -> None:
        pass
