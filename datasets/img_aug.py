import PIL
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImgAug(object):
    def __init__(self, seq: iaa.Sequential):
        """
        使用imgaug库,放在ToTensor前
        :param img: IAA.Sequence
        """
        self.seq = seq

    def __call__(self, img):
        img = np.asarray(img)
        img = self.seq.augment_image(img)
        img = Image.fromarray(img)
        return img


if __name__ == '__main__':
    seq = iaa.Sequential([
        # iaa.Crop(px=(1, 16), keep_size=False),
        # iaa.Fliplr(0.5),
        # iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.1, 0.4)),
    ])
    aug = ImgAug(seq)
    rawimg = Image.open('/home/cmf/Pictures/Screenshot from 2019-10-15 17-50-59.png')
    # rawimg = cv2.resize(rawimg, (22))
    rawimg = rawimg.resize((224, 224))
    for i in range(10):
        img = aug(rawimg)
        # img.show('123')
        plt.imshow(np.asarray(img))
        plt.show()
