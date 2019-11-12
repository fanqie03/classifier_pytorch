import argparse
import copy
import time
import os
import sys

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from backbone import *
from datasets.folder import get_dataset

from tools.misc import *

def parse_args():
    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--config', help='train config file path')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    # Params for scheduler
    parser.add_argument('--scheduler', default='multi-step', type=str,
                        choices=['multi-step', 'cosine', 'autoscale-lr'])
    parser.add_argument('--power', default=2, type=int,
                        help='poly lr pow')
    # Params for Multi-step Scheduler
    parser.add_argument('--milestons', default='30,50', type=str,
                        help='milestons for MultiStepLR')
    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')
    # Params for Autoscale-lr

    # Train params
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--start_epochs', default=0, type=int)

    # Params for optimizer
    parser.add_argument('--optimizer_type', default="SGD", type=str,
                        help='optimizer_type')
    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--head_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')
    # Params for Adam

    # Params for data
    parser.add_argument('--input_size', default=[224, 224], nargs='+', type=int)  # TODO check list arguments
    parser.add_argument('--train_datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--train_datasets_type', default=['ImageFolder'], nargs='+', help='Dataset type')
    parser.add_argument('--validation_datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_datasets_type', default=['ImageFolder'], nargs='+', help='Dataset, type')

    # Params for model
    parser.add_argument('--net_type', default='Net2')
    parser.add_argument('--freeze_body', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_head', action='store_true')
    parser.add_argument('--pretrained_model', default=None,
                        help='start epoch will be 0')
    parser.add_argument('--resume_from', default=None,
                        help='the checkpoint file to resume from, start epoch will be that')
    parser.add_argument('--init_type', default='xavier')
    parser.add_argument('--num_classes', default=2, type=int)

    # Params for loss
    parser.add_argument('--loss_type', default='CrossEntropyLoss')


    # Params for other
    parser.add_argument('--cuda_index', default=['0'], type=str, nargs='+')  # TODO check list arguments
    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    parser.add_argument('--work_dir', default='checkpoint',
                        help='the dir to save logs and models')

    args = parser.parse_args()

    if args.config:
        merge_from_file(args, args.config)

    return args


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = 99999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
    args = parse_args()

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    work_dir = os.path.join(args.work_dir, t)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)


    sys.stdout = Logger(os.path.join(work_dir, 'log.txt'))

    with open(os.path.join(work_dir, 'meta.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            f.write("key:[{}], value:[{}]\r\n".format(key, value))
    print(args)

    device = torch.device(f'cuda:{args.cuda_index[0]}'
                          if torch.cuda.is_available() and args.use_cuda else 'cpu')

    writer = SummaryWriter(os.path.join(work_dir, 'runs'))
    # writer.add_graph()
    # TODO add info after each TODO list
    # TODO net
    model = eval(args.net_type)(num_classes=args.num_classes)
    model = model.to(device)
    print(f'use model {args.net_type}, num_classes is {args.num_classes}')
    # TODO load model
    if args.resume_from:
        m = torch.load(args.resume_from)
        model.load_state_dict(m['model'])
        args.start_epoch = m['epoch']
        optimizer_state_dict = m['optimizer']
        best_score = m['best_score']
        print(f'load model from {args.resume_from}, start_epoch is {args.start_epoch}, best_score is {best_score}')
    if args.pretrained_model:
        m = torch.load(args.pretrained_model)
        model.load_state_dict(m['model'])
        best_score = m['best_score']
        print(f'load model from {args.resume_from}, best_score is {best_score}')

    # TODO freeze
    if args.freeze_body and getattr(model, 'body'):
        model.body.parameters()
        print(f'freeze model body')
    if args.freeze_head and getattr(model, 'head'):
        model.head.parameters()
        print(f'freeze model head')
    # TODO check multigpu
    if args.use_cuda and len(args.cuda_index) > 1:
        model = nn.DataParallel(model, args.cuda_index)
        print(f'use multi gpu {args.cuda_index}')
    # TODO loss
    if args.loss_type == 'CrossEntropyLoss':
        loss = torch.nn.CrossEntropyLoss()
    print(f'use loss type is {args.loss_type}')
    # TODO scheduler
    # scheduler = eval()
    # TODO lr and optimizer
    # if args.optimizer_type ==
    optimizer = eval(args.optimizer_type)()
    # TODO transform

    # TODO datasets
    # TODO dataloader
    # TODO train and test loop


    # global dataloaders, dataset_sizes, class_names
    # dataloaders, dataset_sizes, class_names = get_dataset(
    #     batch_size=args.batch_size, num_workers=args.num_workers,
    #     data_root=args.data_root)
    # if args.net_type.startswith('get_pretrained_net'):
    #     net = eval(args.net_type)
    # else:
    #     net = eval(args.net_type)(len(class_names))
    #
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = nn.DataParallel(net)
    #
    # net.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # # optimizer = optim.Adam(net.parameters(), )
    #
    # if args.pretrained_model:
    #     m = torch.load(args.pretrained_model)
    #     net.load_state_dict(m['model_state_dict'])
    #     print('load pretrained model : ' + args.pretrained_model)
    #     if m.get('optimizer_state_dict') is not None:
    #         optimizer.load_state_dict(m['optimizer_state_dict'])
    #         print('load optimizer')
    #
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #
    # model = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
