from pathlib import Path
import os
import random
import argparse
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description='split data to train set and val set')
parser.add_argument('--test_size', default=0.3)
parser.add_argument('--input_folder', default='/home/cmf/datasets/extract_data/classifier')
parser.add_argument('--output_folder', default='/home/cmf/datasets/extract_data/')
args = parser.parse_args()

input_folder = Path(args.input_folder)
output_folder = Path(args.output_folder)
input_files = list(input_folder.rglob('*/*'))

size = len(input_files)
test_size = int(args.test_size * size)
train_size = size - test_size

train_set, test_set = train_test_split(input_files, test_size=args.test_size)


def cp(data_set, dst_folder):
    """

    :param data_set: 包含文件名
    :param dst_folder: 目标文件夹
    :return:
    """
    for item in tqdm(data_set):
        catagorical = item.parts[-2]
        name = item.name
        output_folder = dst_folder / catagorical
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = output_folder / name
        shutil.copy(str(item), str(output_file))


cp(train_set, output_folder / 'train')
cp(test_set, output_folder / 'val')
