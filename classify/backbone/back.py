import torch.nn as nn
import torch.nn.functional as F
from tools.module import init_weight
import torchvision
from torchvision import models
import torch



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def test_init_model():
    # Initialize the model for this run
    model_name = 'resnet'
    num_classes = 3
    feature_extract = True
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)
    # print(model_ft.require_grad)
    # model_ft.eval()

    for param in model_ft.parameters():
        print(param.shape, param.requires_grad)



def get_pretrained_net(name, num_classes):
    model = eval("torchvision.models." + name)(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def resnet18(num_classes):
    model = torchvision.models.resnet18(True)
    model.fc = nn.Sequential(
        nn.Dropout2d(0.3),
        nn.Linear(in_features=512, out_features=num_classes, bias=True)
    )
    init_weight(model.fc)
    return model


def resnet34(num_classes):
    # model = torchvision.models.resnet34(34)
    # model.fc = nn.
    pass


def down_sam_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
    )


class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super(TinyNet, self).__init__()
        self.model = nn.Sequential(
        down_sam_blk(3, 16),
        down_sam_blk(16, 32),
        down_sam_blk(32, 64),
        down_sam_blk(64, 128),
        nn.Conv2d(128, num_classes, 1, 1, 0),
        nn.AdaptiveAvgPool2d(1)
        )
        init_weight(self.model)

    def forward(self, x):
        x = self.model(x)
        x = torch.reshape(x, (x.size(0), -1))
        return x


def test_tiny_net():
    model = TinyNet(2)
    x = torch.ones((1, 3, 224, 224))
    r = model(x)
    print(r.shape)


def test_resnet18():
    model = resnet18(3)
    print(model)


class SimpleNet(nn.Module):

    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
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
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
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


def squeezenet1_1(num_classes):
    model = torchvision.models.squeezenet1_1(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1))
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
    # get_pretrained_net('resnet50', 10)
    test_tiny_net()