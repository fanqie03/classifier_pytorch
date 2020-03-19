import torch
from torch import nn


class AllconvNetDepth5(nn.Module):
    def __init__(self, num_classes):
        super(AllconvNetDepth5, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.Conv2d(64, 64, 3, 2),
            nn.Conv2d(64, 128, 3, 2),
            nn.Conv2d(128, 128, 3, 2),
            nn.Conv2d(128, 256, 3, 2),
        )
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class AllconvNetDepth6(nn.Module):
    def __init__(self, num_classes):
        super(AllconvNetDepth6, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.Conv2d(64, 64, 3, 2),
            nn.Conv2d(64, 128, 3, 2),
            nn.Conv2d(128, 128, 3, 2),
            nn.Conv2d(128, 256, 3, 2),
        )
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class AllconvNetDepth7(nn.Module):
    def __init__(self, num_classes):
        super(AllconvNetDepth7, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.Conv2d(64, 64, 3, 2),
            nn.Conv2d(64, 128, 3, 2),
            nn.Conv2d(128, 128, 3, 2),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.Conv2d(128, 256, 3, 2),
        )
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

if __name__ == '__main__':
    data = torch.ones((1, 3, 224, 224))
    # models = [
    #     AllconvNet0_25(10),
    #     AllconvNet0_5(10),
    #     AllconvNet0_75(10),
    #     AllconvNet1(10),
    #     AllconvNet1_25(10),
    #     AllconvNet1_5(10),
    #     AllconvNet1_75(10),
    # ]
    #
    # for model in models:
    #     print(model)
    #     r = model(data)
    #     print(r.shape)
