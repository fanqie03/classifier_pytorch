import os

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from datasets.img_aug import ImgAug
# import albumentations as


def get_dataset(size=(224, 224),
                data_root='/home/cmf/datasets/helmet_all/train_val',
                batch_size=32, num_workers=4,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]):
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.5,
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.1, 0.4)),
            # iaa.JpegCompression()
        ),
        # iaa.MotionBlur(k=13)
        # iaa.GaussianBlur((0, 3.0)),
    ])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
            transforms.RandomAffine(50),
            # transforms.Grayscale(3),
            ImgAug(seq),
            transforms.ToTensor(),
            transforms.RandomErasing(ratio=(10, 15), inplace=True),
            transforms.RandomErasing(ratio=(10, 15), inplace=True),
            transforms.RandomErasing(ratio=(0.1, 0.2), inplace=True),
            transforms.RandomErasing(ratio=(0.1, 0.2), inplace=True),
            # transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    }

    image_datasets = {
        x: ImageFolder(os.path.join(data_root, x),
                       data_transforms[x]) for x in ('train', 'val')}
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size,
                      shuffle=True, num_workers=num_workers) for x in ('train', 'val')
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ('train', 'val')}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    dataloaders, dataset_sizes, class_names = get_dataset()
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    print(out)
    imshow(out, title=[class_names[x] for x in classes])
