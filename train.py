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

from tools import Logger


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

    parser.add_argument('--data_root', default='/home/cmf/datasets/helmet_all/train_val')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', default=6)
    parser.add_argument('--total_epochs', default=700)
    # parser.add_argument('--net_type', default='get_pretrained_net("resnet50", 3)')
    parser.add_argument('--net_type', default='Net2')
    parser.add_argument('--pretrained_model', default='checkpoint/2019-10-22 12:58:25/model_resnet18_8_0.0026_0.9996.pth')
    args = parser.parse_args()
    return args


def train_model(model, criterion, optimizer, scheduler, total_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = 99999999

    for epoch in range(total_epochs):
        print('Epoch {}/{}'.format(epoch, total_epochs - 1))
        print('-' * 10)
        # print('learning rate is {}'.format(optimizer))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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

            scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            writer.add_scalar('{} Loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('{} Acc'.format(phase), epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and (epoch_acc > best_acc or epoch_loss < min_loss):
                best_acc = epoch_acc
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model_pth = os.path.join(work_dir,
                                         'model_{}_{}_{:.4f}_{:.4f}.pth'.format(args.net_type, epoch, epoch_loss,
                                                                                epoch_acc))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'classes': class_names
                }, model_pth)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, min loss {:4f}'.format(best_acc, min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    global args, work_dir
    args = parse_args()

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    work_dir = os.path.join(args.work_dir, t)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    import sys
    sys.stdout = Logger(os.path.join(work_dir, 'log.txt'))

    with open(os.path.join(work_dir, 'meta.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            f.write("key:[{}], value:[{}]\r\n".format(key, value))

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

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), )

    if args.pretrained_model:
        m = torch.load(args.pretrained_model)
        net.load_state_dict(m['model_state_dict'])
        print('load pretrained model : ' + args.pretrained_model)
        if m.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(m['optimizer_state_dict'])
            print('load optimizer')

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(net, criterion, optimizer, exp_lr_scheduler, total_epochs=args.total_epochs)


if __name__ == '__main__':
    main()
