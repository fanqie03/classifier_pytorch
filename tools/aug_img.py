from albumentations import *
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def strong_aug(p=0.5):
    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        # 天气
        OneOf([
            RandomRain(),
            # RandomSunFlare(),
            RandomFog(),
            RandomBrightness(),
            RandomShadow(),
            RandomSnow(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('output_path')
    parser.add_argument('--aug_loop', type=int, default=1)
    parser.add_argument('--aug_count_per_image', type=int, default=1000)
    return parser.parse_args()


def main():
    args = get_args()
    root = Path(args.image_path)
    output_path = Path(args.output_path)
    imgs = list(root.rglob('*'))
    augmentation = strong_aug(p=0.9)
    print(len(imgs))
    for img_file in imgs:
        name = img_file.name.split('.')[0]
        img = np.asarray(Image.open(str(img_file)))[:, :, 0:3]
        data = {"image": img}
        for i in tqdm(range(args.aug_count_per_image)):
            augmented = augmentation(**data)
            aug_img = augmented['image']
            aug_img = Image.fromarray(aug_img)
            aug_img.save("{}/{}_{}.jpg".format(str(output_path), name, i))


if __name__ == '__main__':
    main()
