import numpy as np
import os
from pathlib import Path
from PIL import Image


class GHIMDataset:
    def __init__(self, root, transform=None):
        # root = Path(root)
        # self.imgs_path = list(root.rglob('*.jpg'))
        # self.labels = [int(x.name.split('_')[0]) for x in self.imgs_path]

        self.imgs_path = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('.jpg')]
        self.labels = [int(x.split('/')[-1].split('_')[0]) for x in self.imgs_path]
        self.classes = set(self.labels)
        self.num_classes = len(self.classes)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(str(self.imgs_path[index]))
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    dataset = GHIMDataset('/home/cmf/datasets/Corel/Corel100└α┐Γ')
    print(len(dataset))
    print(dataset.__getitem__(0))
    print(dataset.classes)
    print(dataset.num_classes)
    for img, label in dataset:
        print(img, label)
