import torch.nn as nn
import torch.nn.functional as F
from tools.module import init_weight
import torchvision
import torch


def get_pretrained_net(name, num_classes):
    model = eval("torchvision.models." + name)(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def resnet18(num_classes):
    model = torchvision.models.resnet18(True)
    model.fc = nn.Sequential(
        nn.Dropout2d(0.7),
        nn.Linear(in_features=512, out_features=num_classes, bias=True)
    )
    init_weight(model.fc)
    return model


def test_resnet18():
    model = resnet18(3)
    print(model)


class LeNet(nn.Module):
    """
    # recommend input_size is 32x32
    """

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)  # 6x28x28
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6x14x14
        # self.conv3 = nn.Conv2d(6, 16, 5)  # 16x10x10
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x5x5
        # self.conv5 = nn.Conv2d(16, 120, 5)  # 120x1x1
        # self.fc6 = nn.Linear(120, 84)
        # self.fc7 = nn.Linear(84, num_classes)

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test_lenet():
    data = torch.ones((1, 1, 32, 32))
    model = LeNet(10)
    model.eval()
    print(model)
    ret = model(data)
    print(ret.size())


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
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.pool = nn.MaxPool2d(2) if pool else None
        init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm is not None else x
        x = eval("F." + self.activation)(x, inplace=True) if self.activation is not None else x
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
