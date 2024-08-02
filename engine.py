import torch
import torch.distributed as dist
import tqdm
from torchvision.transforms import v2

def train_one_epoch(
    train_sampler,
    trainloader,
    opt,
    ddp_model,
    objective,
    scaler,
    epoch,
    device_id,
    writer,
):
    mixup = v2.MixUp(num_classes=10)

    ddp_model.train()
    train_sampler.set_epoch(epoch)
    for step, (data, labels) in enumerate(trainloader):
        opt.zero_grad()

        data = data.to(device_id)
        labels = labels.to(device_id)

        data, labels = mixup(data, labels)

        with torch.autocast(device_type='cuda'):
            pred = ddp_model(data)
            loss = objective(pred, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % 10:
            log_str = f'Epoch: {epoch}, Iter: {step}/{len(trainloader)}, Loss: {loss.cpu().item()}'
            print(log_str)

@torch.no_grad()
def test(
    ddp_model,
    testloader,
    device_id,
    epoch,
    writer,
):
    ddp_model.eval()

    total = 0.
    correct = 0.
    for data, labels in tqdm.tqdm(testloader):
        data = data.to(device_id)
        labels = labels.to(device_id)

        with torch.autocast(device_type='cuda'):
            pred = ddp_model(data)
            _, class_pred = torch.max(pred, 1)

        gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, labels)
        gathered_labels = torch.cat(gathered_labels)

        gathered_class_preds = [torch.zeros_like(class_pred) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_class_preds, class_pred)
        gathered_class_preds = torch.cat(gathered_class_preds)

        total += gathered_labels.size(0)
        correct += (gathered_class_preds == gathered_labels).sum().item()

    accuracy = correct / total
    log_str = f'Evaluation --------- Epoch: {epoch}, Accuracy: {accuracy}'
    print(log_str)
    writer.add_scalar('Eval/Accuracy', accuracy, epoch)
