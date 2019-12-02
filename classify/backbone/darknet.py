import torch
from torch import nn

class YOLOv3TinyBackbone(nn.Module):
    # TODO check Conv type
    def __init__(self, num_class):
        super(YOLOv3TinyBackbone, self).__init__()
        # 416x416
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, 1), # 13x13
            # nn.MaxPool2d(2, 2),
        )

        self.neck = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
