import argparse
import copy
import time
import os

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from backbone import *
from datasets.folder import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--config', help='train config file path', default='configs/helmet.py')
    parser.add_argument('--work_dir', default='checkpoint', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    parser.add_argument('--data_root', default='/home/cmf/datasets/extract_data')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--total_epochs', default=50)
    # parser.add_argument('--net_type', default='get_pretrained_net("resnet50", 3)')
    parser.add_argument('--net_type', default='Net2')
    args = parser.parse_args()
    return args


def train_model(model, criterion, optimizer, scheduler, total_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(total_epochs):
        print('Epoch {}/{}'.format(epoch, total_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            writer.add_scalar('{} Loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('{} Acc'.format(phase), epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    work_dir = os.path.join(args.work_dir, t)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    global writer
    writer = SummaryWriter(os.path.join(work_dir, 'runs'))

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    global dataloaders, dataset_sizes, class_names
    dataloaders, dataset_sizes, class_names = get_dataset(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root)
    if args.net_type.startswith('get_pretrained_net'):
        net = eval(args.net_type)
    else:
        net = eval(args.net_type)(len(class_names))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(net, criterion, optimizer, exp_lr_scheduler, total_epochs=args.total_epochs)

    model_pth = os.path.join(work_dir, 'model_{}.pth'.format(args.net_type))

    torch.save({
        'epoch': args.total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': class_names
    }, model_pth)


if __name__ == '__main__':
    main()
