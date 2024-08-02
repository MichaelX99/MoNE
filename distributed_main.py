from MoNE import MoNE
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import builtins
from engine import train_one_epoch, test
from torch.utils.tensorboard import SummaryWriter
import typer

def fix_print(rank):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)
    builtins.print = print


def main(
    eps: float = 0.6, # NOTE capacity is randomized during training, only used at test time # TODO implement test time
    batch_size: int = 256, # total effective batch size for the world
    num_epochs: int = 50,
):
    # TODO
    # add in learning rate scheduler
    # add in stochastic depth regularization
    # add in warm up
    assert eps <= 1.0 and eps >= 0., f'eps must be between 0 and 1, not {eps}'

    dist.init_process_group('nccl')
    rank = dist.get_rank()
    print(f'Starting DDP on rank {rank}')
    fix_print(rank)

    train_transform = transforms.Compose(
        [
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    batch_size = batch_size // dist.get_world_size()
    print(f'Using per-gpu batch size: {batch_size}')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_sampler = torch.utils.data.DistributedSampler(trainset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=batch_size, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset, num_workers=8, batch_sampler=train_batch_sampler)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_sampler = torch.utils.data.DistributedSampler(testset, shuffle=False)
    test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, batch_size=batch_size, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, num_workers=8, batch_sampler=test_batch_sampler)

    img_shape = trainset[0][0].shape[-1]
    device_id = rank % torch.cuda.device_count()
    model = MoNE(img_shape, eps).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    objective = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(ddp_model.parameters())
    scaler = torch.amp.GradScaler(device_id)

    writer = SummaryWriter()
    
    print('Starting training')
    for epoch in range(num_epochs):
        train_one_epoch(
            train_sampler,
            trainloader,
            opt,
            ddp_model,
            objective,
            scaler,
            epoch,
            device_id,
            writer,
        )

        test(
            ddp_model,
            testloader,
            device_id,
            epoch,
            writer,
            eps,
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    typer.run(main)
