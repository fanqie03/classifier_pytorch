import argparse
import copy
import time
import os
import sys
import copy
import subprocess

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
from tools.builder import build_transform

import random
from tools.misc import *


def parse_args():
    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--config', help='train config file path', default='configs/defaults.yaml')

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


def train(model, criterion, optimizer, loader, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num = 0
    for i, data in enumerate(loader):
        num += 1
        # if random.random() > 0.8:
        # print('\r' + '.' * (i + 1), end='')
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        score = model(images)
        # TODO check softmax
        loss = criterion(score, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(score, 1)
        acc = torch.sum(preds == labels.data).detach().cpu().numpy() * 1.0
        acc = acc / len(images)
        loss = loss.item()

        step = epoch * len(loader) + i
        writer.add_scalar('train_loss', loss, global_step=step)
        writer.add_scalar('train_acc', acc, global_step=step)

        running_acc += acc
        running_loss += loss

    running_acc = running_acc / num
    running_loss = running_loss / num

    # step = (epoch + 1) * len(loader)
    # writer.add_scalar('train_loss', running_loss, global_step=step)
    # writer.add_scalar('train_acc', running_acc, global_step=step)

    return running_loss, running_acc


def test(model, criterion, loader, device, global_step, writer=None, ):
    model.eval()
    running_loss = 0.0
    running_acc = 0
    num = 0
    for i, data in enumerate(loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            score = model(images)
            loss = criterion(score, labels)
            _, preds = torch.max(score, 1)

            acc = torch.sum(preds == labels.data).detach().cpu().numpy() * 1.0
            acc = acc / len(images)
            loss = loss.item()
            running_acc += acc
            running_loss += loss

    running_loss /= num
    running_acc /= num

    writer.add_scalar('val_loss', running_loss, global_step=global_step)
    writer.add_scalar('val_acc', running_acc, global_step=global_step)

    return running_loss, running_acc


def main():
    cfg = parse_args()
    _cfg = copy.deepcopy(cfg)

    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    work_dir = os.path.join(cfg.train.work_dir, t)
    ckpt_dir = os.path.join(work_dir, 'ckpt')
    model_dir = os.path.join(work_dir, 'model')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sys.stdout = Logger(os.path.join(work_dir, 'log'))

    with open(os.path.join(work_dir, 'meta.txt'), 'w') as f:
        pprint(cfg, f)
    print(cfg)

    device = torch.device(f'cuda:{cfg.device.cuda_index[0]}'
                          if torch.cuda.is_available() and cfg.device.use_cuda else 'cpu')

    writer_dir = os.path.join(work_dir, 'runs')
    writer = SummaryWriter(writer_dir, flush_secs=1)
    # writer.add_hparams(cfg.__dict__)
    if cfg.train.open_tensorboard:
        subprocess.Popen(f'tensorboard --logdir {writer_dir}', shell=True)
    # writer.f
    #  net
    print(f'use model {cfg.model}')
    model = eval(cfg.model.pop('type'))(**cfg.model)
    demo_data = torch.ones([1, 3] + cfg.data.input_size)
    if cfg.get('data') and cfg.data.get('input_size'):
        writer.add_graph(model, torch.ones([1, 3] + cfg.data.input_size))
        print(f'Add Graph')

    # onnx_path = os.path.join(work_dir, f'{_cfg.model.type}.onnx')
    # torch.onnx.export(model, demo_data, onnx_path)
    # writer.add_onnx_graph(onnx_path)

    with open(os.path.join(work_dir, 'model.txt'), 'w') as f:
        print(model, file=f)

    #  load model
    if cfg.train.resume_from:
        print(f'resume model {cfg.train.resume_from}')
        m = torch.load(cfg.train.resume_from)
        model.load_state_dict(m['model'])
        cfg.train.start_epoch = m['epoch']
        optimizer_state_dict = m['optimizer']
        best_score = m['best_score']
        print(
            f'resume model from {cfg.train.resume_from}, start_epoch is {cfg.train.start_epoch}, best_score is {best_score}')
    if cfg.train.pretrained_model:
        print(f'pretrained model {cfg.train.pretrained_model}')
        m = torch.load(cfg.train.pretrained_model)
        model.load_state_dict(m['model'])
        best_score = m['best_score']
        print(f'load pretrained model from {cfg.train.pretrained_model}, best_score is {best_score}')

    model = model.to(device)

    #  freeze
    if cfg.train.freeze_body and getattr(model, 'body'):
        model.body.parameters()
        print(f'freeze model body')
    if cfg.train.freeze_head and getattr(model, 'head'):
        model.head.parameters()
        print(f'freeze model head')
    #  check multigpu
    if cfg.device.use_cuda and len(cfg.device.cuda_index) > 1:
        model = nn.DataParallel(model, cfg.device.cuda_index)
        print(f'use multi gpu {cfg.device.cuda_index}')
    #  loss
    print(f'use loss {cfg.loss}')
    loss = eval(cfg.loss.pop('type'))(**cfg.loss)
    #  lr and optimizer
    cfg.optimizer.params = model.parameters()
    optimizer = eval(cfg.optimizer.pop('type'))(**cfg.optimizer)
    #  scheduler
    cfg.scheduler.optimizer = optimizer
    scheduler = eval(cfg.scheduler.pop('type'))(**cfg.scheduler)
    #  transform
    transform = target_transform = None
    val_transform = val_target_transform = None
    transform = val_transform = build_transform(cfg.transform)
    #  datasets

    for d in cfg.train_datasets:
        d.transform = transform
        d.target_transform = target_transform

    dataset = ConcatDataset([eval(c.pop('type'))(**c) for c in cfg.train_datasets])
    print(f'train dataset classes is {[getattr(dataset, "classes") for dataset in dataset.datasets]}')
    classes = dataset.datasets[0].classes
    print(f'total train datasets is {len(dataset)}')
    for d in cfg.val_datasets:
        d.transform = val_transform
        d.target_transform = val_target_transform
    val_dataset = ConcatDataset([eval(c.pop('type'))(**c) for c in cfg.val_datasets])
    print(f'total validation datasets is {len(val_dataset)}')
    print(f'validation dataset classes is {[getattr(dataset, "classes") for dataset in val_dataset.datasets]}')
    #  dataloader
    cfg.dataloader.dataset = dataset
    dataloader = DataLoader(**cfg.dataloader)
    cfg.dataloader.dataset = val_dataset
    val_dataloader = DataLoader(**cfg.dataloader)
    #  train and test loop
    opt = cfg.train
    print(f'Start training from epoch {opt.start_epoch}.')
    for epoch in range(opt.start_epoch, opt.num_epochs):
        # 学习率调整
        if epoch != opt.start_epoch:
            scheduler.step()
        print(f'lr rate :{optimizer.param_groups[0]["lr"]}')
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch * len(dataloader))

        train_loss, train_acc = train(model, loss, optimizer, dataloader, device, epoch, writer)

        if epoch % opt.val_epochs == 0 or epoch == opt.num_epochs - 1:
            print(f'lr rate :{optimizer.param_groups[0]["lr"]}')

            val_loss, val_acc = test(model, loss, val_dataloader, device, (epoch + 1) * len(dataloader), writer)

            print(
                f'Epoch: {epoch}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.4f}, '
                f'Validation Loss: {val_loss:.4f}, '
                f'Validation Accuracy: {val_acc:.4f}'
            )

            ckpt_path = os.path.join(ckpt_dir, f'{_cfg.model.type}-Epoch-{epoch}-Loss-{val_loss}.pth')
            model_path = os.path.join(model_dir, f'{_cfg.model.type}-Epoch-{epoch}-Loss-{val_loss}.pth')

            score = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }

            save_checkpoint(epoch, model.state_dict(), scheduler.state_dict(), score, classes, ckpt_path, model_path)
            print(f'Saved model {model_path}')
            print(f'Saved checkpoint {ckpt_path}')


if __name__ == '__main__':
    main()
