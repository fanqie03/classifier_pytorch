import torch
from torch import nn


class AllconvNetChannels(nn.Module):
    def __init__(self, num_classes, channels_percent=1.):
        super(AllconvNetChannels, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, int(64 * channels_percent), 3, 2),
            nn.Conv2d(int(64 * channels_percent), int(64 * channels_percent), 3, 2),
            nn.Conv2d(int(64 * channels_percent), int(128 * channels_percent), 3, 2),
            nn.Conv2d(int(128 * channels_percent), int(128 * channels_percent), 3, 2),
            nn.Conv2d(int(128 * channels_percent), int(256 * channels_percent), 3, 2),
        )
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(int(256 * channels_percent), num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def AllconvNet0_25(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=0.25)


def AllconvNet0_5(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=0.5)


def AllconvNet0_75(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=0.75)


def AllconvNet1(num_classes):
    return AllconvNetChannels(num_classes)


def AllconvNet1_25(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=1.25)


def AllconvNet1_5(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=1.5)


def AllconvNet1_75(num_classes):
    return AllconvNetChannels(num_classes, channels_percent=1.75)


if __name__ == '__main__':
    data = torch.ones((1, 3, 224, 224))
    models = [
        AllconvNet0_25(10),
        AllconvNet0_5(10),
        AllconvNet0_75(10),
        AllconvNet1(10),
        AllconvNet1_25(10),
        AllconvNet1_5(10),
        AllconvNet1_75(10),
    ]

    for model in models:
        print(model)
        r = model(data)
        print(r.shape)
