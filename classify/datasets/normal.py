import torch
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
import cv2
from PIL import Image
import time

class ImageDataset(Dataset):

    def __init__(self, image_path, transform=None):
        p = Path(image_path)
        self.imgs = [str(x) for x in p.glob('*')]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor()
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir')
    parser.add_argument('--image_size', type=int, default=50)
    parser.add_argument('--ckpt', default='ckpt')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # dataset = ImageDataset(args.image_dir, get_transform(args))
    dataset = ImageDataset(args.image_dir)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    for index in range(len(dataset)):
        img, img = dataset[index]
        img.show('1')
        time.sleep(1)


if __name__ == '__main__':
    main()
