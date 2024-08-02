from MoNE import MoNE
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import builtins

def fix_print(rank):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)
    builtins.print = print


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    print(f'Starting DDP on rank {rank}')
    fix_print(rank)


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    batch_size = 256 // dist.get_world_size()
    print(f'Using per-gpu batch size: {batch_size}')
    num_epochs = 20

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.DistributedSampler(trainset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset, num_workers=8, batch_sampler=batch_sampler)

    img_shape = trainset[0][0].shape[-1]
    device_id = rank % torch.cuda.device_count()
    model = MoNE(img_shape).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    objective = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(ddp_model.parameters())
    scaler = torch.amp.GradScaler(device_id)

    alpha = 0.
    print('Starting training')
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for ind, (data, labels) in enumerate(trainloader):
            opt.zero_grad()

            data = data.to(device_id)
            labels = labels.to(device_id)

            with torch.autocast(device_type='cuda'):
                pred = ddp_model(data, alpha)
                loss = objective(pred, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if ind % 10:
                log_str = f'Epoch: {epoch}, Iter: {ind}/{len(trainloader)}, Loss: {loss.cpu().item()}'
                print(log_str)

        alpha += 0.1



    dist.destroy_process_group()

if __name__ == "__main__":
    main()
