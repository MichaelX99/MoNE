from MoNE import MoNE
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    batch_size = 256
    num_epochs = 20

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    img_shape = trainset[0][0].shape[-1]
    model = MoNE(img_shape).to('cuda')

    objective = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler('cuda')

    # TODO
    # add in model checkpointing
    # add in distributed training

    alpha = 0.
    print('Starting training')
    for epoch in range(num_epochs):
        for ind, (data, labels) in enumerate(trainloader):
            opt.zero_grad()

            data = data.to('cuda')
            labels = labels.to('cuda')

            with torch.autocast(device_type='cuda'):
                pred = model(data, alpha)
                loss = objective(pred, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if ind % 10:
                log_str = f'Epoch: {epoch}, Iter: {ind}/{len(trainloader)}, Loss: {loss.cpu().item()}'
                print(log_str)

        alpha += 0.1

if __name__ == "__main__":
    main()