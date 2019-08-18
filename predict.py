import torch
import torchvision
import argparse
from backbone import *
import shutil
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tools.files import check_dir
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='checkpoint/2019-08-16 15:03:22/model_mobilenet_v2.pth', help='path of saved model file')
    parser.add_argument('--model', default='mobilenet_v2', help='model type')
    parser.add_argument('--input_dir', default='/home/cmf/datasets/extract_data/gongdi/person', help='the directory of test')
    parser.add_argument('--output_dir', default='/home/cmf/datasets/extract_data/test_predict', help='the directory of predict result')
    parser.add_argument('--image_size', default=224)
    parser.add_argument('--threshold', type=float, help='')
    args = parser.parse_args()
    return args

def preprocess_image(image):
    o = args.image_size
    image = image.resize((o, o), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32)
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
        image = preprocess_image(image)
        image = torch.from_numpy(image).to(device)
        result = model(image)
        _, preds = torch.max(result, dim=1)
        index = preds.cpu().numpy()
        class_name = classes[index[0]]
        output_path = output_dir / class_name
        check_dir(output_path)
        output_file = output_path / file.name
        shutil.copy(file, output_file)
        # print(_, preds, class_name)







if __name__ == '__main__':
    main()