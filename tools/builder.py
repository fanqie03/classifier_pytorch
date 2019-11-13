import torch
from torchvision.transforms.transforms import *


def build_transform(cfg):
    return Compose([
        Resize((224, 224)),
        ToTensor()
    ])