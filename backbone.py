import torch.nn as nn
import torch.nn.functional as F
from tools.module import init_weight
import torchvision


def get_pretrained_net(name, num_classes):
    model = eval("torchvision.models." + name)(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def mobilenet_v2(num_classes):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout2d(p=0.3, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    init_weight(model.classifier)
    return model

class Net(nn.Module):
    """
    Best val Acc: 0.777027
    """

    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        init_weight(self)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        # print(x.shape)
        batch_size, channel, height, width = x.shape
        x = x.view(-1, channel * height * width)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm=True, activation='relu', pool=True):
        super(BasicConv2d, self).__init__()
        self.activation=activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.pool = nn.MaxPool2d(2) if pool else None
        init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm is not None else x
        x = eval("F."+self.activation)(x, inplace=True) if self.activation is not None else x
        x = self.pool(x) if self.pool is not None else x

        return x


class Net2(nn.Module):
    """
    Best val Acc: 0.777027
    """

    def __init__(self, num_classes):
        super(Net2, self).__init__()
        self.conv1_1 = BasicConv2d(3, 16)
        # self.conv1_2 = BasicConv2d(16, 16, pool=False)
        self.conv2 = BasicConv2d(16, 32)
        self.conv3 = BasicConv2d(32, 64)
        self.conv4 = BasicConv2d(64, 128)
        self.fc1 = nn.Linear(128 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        init_weight(self)

    def forward(self, x):
        x = self.conv1_1(x)
        # x = self.conv1_2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        batch_size, channel, height, width = x.shape
        x = x.view(-1, channel * height * width)
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

class Net1(nn.Module):
    def __init__(self, num_classes):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.norm5 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        init_weight(self)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.pool(F.relu(self.norm4(self.conv4(x))))
        x = self.pool(F.relu(self.norm5(self.conv5(x))))
        x = self.global_pool(x)
        batch_size, channel, height, width = x.shape
        # print(x.shape)
        x = x.view(-1, channel * height * width)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_net():
    net = Net(num_classes=10)
    print(net)


def test_get_pretrained_net():
    print(get_pretrained_net('resnet50', 10))


if __name__ == '__main__':
    get_pretrained_net('resnet50', 10)
