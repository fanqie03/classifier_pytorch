from torch import nn
import torch
from torchvision.transforms import *
from torch.utils.data import DataLoader
from PIL import Image
from classify.utils.misc import to_img
from classify.datasets.normal import ImageDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    distance = nn.MSELoss()
    num_epochs = 100  # you can go for more epochs, I am using a mac
    batch_size = 128
    img_size = 50
    show_interval = 5
    img_ = Image.open('/home/cmf/Pictures/楼顶右边704x576.png')
    model = Autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor()])
    dataset = ImageDataset('/home/cmf/share/Hardhat/Test/JPEGImage', transform)
    dataloader = DataLoader(dataset, batch_size, num_workers=8)

    # img = transform(img_)
    # img = img.unsqueeze(0)
    # img = img.cuda()
    for epoch in range(num_epochs):
        model.train()
        for img in dataloader:
            img = img.cuda()
            # label = label.cuda()
            # ===================forward=====================
            output = model(img)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))
        if (epoch + 1) % show_interval == 0:
            model.eval()
            output = model(img)
            img = np.concatenate([to_img(img, False), to_img(output, False)], axis=1)
            plt.imshow(img)
            plt.show()