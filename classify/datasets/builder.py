import matplotlib.pyplot as plt
from torchvision.transforms.transforms import *
from easydict import EasyDict as edict
from classify.utils.misc import *

from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from classify import get_default_args
from torchvision.datasets import ImageFolder


def build(cfg):
    return eval(cfg.pop('type'))(**cfg)

def build_list(cfgs):
    return [build(cfg) for cfg in cfgs]

def build_transform(transforms):
    transforms = build_list(transforms)
    transform = Compose(transforms)
    return transform

def build_datasets(cfg):

    transform = build_transform(cfg.transform)
    for i, d in enumerate(cfg.train_datasets):
        d.transform = transform
    datasets = build_list(cfg.train_datasets)
    datasets = ConcatDataset(datasets)
    return datasets


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
    cfg = get_default_args()
    # transforms = build_transforms(cfg.transform)
    datasets = build_datasets(cfg)
    loader = DataLoader(datasets, batch_size=32, num_workers=1)
    for i, (images, labels) in enumerate(loader):
        # print(i)
        out = torchvision.utils.make_grid(images)
        print(i, images.shape, labels)
        imshow(out)
        break