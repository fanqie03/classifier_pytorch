import torch
import torchvision
import argparse
from classify.backbone import *
import shutil
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tools.files import check_dir
from tqdm import tqdm
import cv2


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='checkpoint/2019-10-22 15:13:09/model_resnet18_4_0.0022_0.9994.pth',
                        help='path of saved model file')
    parser.add_argument('--model', default='resnet18', help='model type')
    parser.add_argument('--input_dir', default='/home/cmf/datasets/helmet_all/temp1',
                        help='the directory of test')
    parser.add_argument('--output_dir', default='/home/cmf/datasets/helmet_all/predict',
                        help='the directory of predict result')
    parser.add_argument('--image_size', default=224)
    parser.add_argument('--threshold', type=float, help='')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    return args


def preprocess_image(image):
    o = args.image_size
    image = image.resize((o, o), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = image / 255
    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, 0)
    return image


def main():
    global args
    args = get_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dic = torch.load(args.model_path)
    classes = dic['classes']
    model = eval(args.model)(len(classes))
    model.load_state_dict(dic['model_state_dict'])
    model.to(device)
    model.eval()

    files = list(input_dir.rglob('*'))
    for file in tqdm(files):
        image = Image.open(file)
        raw_image = np.array(image)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        draw_image = cv2.resize(raw_image, (448, 448))
        image = preprocess_image(image)
        image = torch.from_numpy(image).to(device)
        result = model(image)
        _, preds = torch.max(result, dim=1)
        index = preds.cpu().numpy()
        class_name = classes[index[0]]
        if args.show:
            cv2.putText(raw_image, class_name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
            # cv2.imshow('demo', raw_image)
            # cv2.waitKey()
        output_path = output_dir / class_name
        check_dir(output_path)
        output_file = output_path / file.name
        # shutil.copy(file, output_file)
        cv2.imwrite(str(output_file), raw_image)
        # print(_, preds, class_name)


if __name__ == '__main__':
    main()
