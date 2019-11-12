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
from easydict import EasyDict as edict
from pprint import pprint
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

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

    args = parser.parse_args()

    args = edict(args.__dict__)

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
    cfg = parse_args()

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    work_dir = os.path.join(cfg.train.work_dir, t)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    sys.stdout = Logger(os.path.join(work_dir, 'log.txt'))

    with open(os.path.join(work_dir, 'meta.txt'), 'w') as f:
        pprint.pprint(cfg, f)
    print(cfg)

    device = torch.device(f'cuda:{cfg.device.cuda_index[0]}'
                          if torch.cuda.is_available() and cfg.device.use_cuda else 'cpu')

    writer = SummaryWriter(os.path.join(work_dir, 'runs'))
    # writer.add_graph()
    # TODO add info after each TODO list
    # TODO net
    print(f'use model {cfg.model}')
    model = eval(cfg.model.pop('type'))(**cfg.model)
    model = model.to(device)
    # TODO load model

    if cfg.train.resume_from:
        print(f'resume model {cfg.train.resume_from}')
        m = torch.load(cfg.train.resume_from)
        model.load_state_dict(m['model'])
        cfg.train.start_epoch = m['epoch']
        optimizer_state_dict = m['optimizer']
        best_score = m['best_score']
        print(f'load model from {cfg.train.resume_from}, start_epoch is {cfg.train.start_epoch}, best_score is {best_score}')
    if cfg.train.pretrained_model:
        print(f'pretrained model {cfg.train.pretrained_model}')
        m = torch.load(cfg.train.pretrained_model)
        model.load_state_dict(m['model'])
        best_score = m['best_score']
        print(f'load model from {cfg.train.resume_from}, best_score is {best_score}')

    # TODO freeze
    if cfg.train.freeze_body and getattr(model, 'body'):
        model.body.parameters()
        print(f'freeze model body')
    if cfg.train.freeze_head and getattr(model, 'head'):
        model.head.parameters()
        print(f'freeze model head')
    # TODO check multigpu
    if cfg.device.use_cuda and len(cfg.device.cuda_index) > 1:
        model = nn.DataParallel(model, cfg.device.cuda_index)
        print(f'use multi gpu {cfg.device.cuda_index}')
    # TODO loss
    print(f'use loss {cfg.loss}')
    loss = eval(cfg.loss.pop('type'))(**cfg.loss)
    # TODO scheduler
    # scheduler = eval()
    # TODO lr and optimizer
    optimizer = eval(cfg.optimizer.pop('type'))(**cfg.optimizer)
    # TODO transform
    transform=target_transform=None
    # TODO datasets
    cfg.train_datasets['transform'] =transform
    cfg.train_datasets['target_transform'] = target_transform
    dataset = ConcatDataset([eval(c.pop('type')(**c)) for c in cfg.train_datasets])
    cfg.validation_datasets['transform'] = transform
    cfg.validation_datasets['target_transform'] = target_transform
    val_dataset = [eval(c.pop('type')(**c)) for c in cfg.validation_datasets]
    # TODO dataloader
    cfg.dataloader['dataset'] = dataset
    dataloader = DataLoader(**cfg.dataloader)
    cfg.dataloader['dataset'] = val_dataset
    val_dataloader = DataLoader(**cfg.dataloader)
    # TODO train and test loop
    # hook.before_train()
    # for iter in range(start_iter, max_iter):
    #     hook.before_step()
    #     trainer.run_step()
    #     hook.after_step()
    # hook.after_train()


if __name__ == '__main__':
    main()
