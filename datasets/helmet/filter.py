import shutil
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='/home/cmf/datasets/extract_data/gongdi/person')
    parser.add_argument('--target_dir', default='/home/cmf/datasets/extract_data/gongdi/person_filter')
    parser.add_argument('--min_size', default=30)
    args = parser.parse_args()
    return args


def split(array, split_num):
    total_len = len(array)
    part_num = len(array) / split_num
    arrays = []
    for i in range(split_num):
        start = int(i*part_num)
        end = int((i+1)*part_num) if int((i+1)*part_num) < total_len else total_len
        arrays.append(array[start: end])
    return arrays

def main():
    args = get_args()
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    source_files = list(source_dir.rglob('*'))
    folder_name = source_dir.name
    for i, file in tqdm(enumerate(source_files)):
        image = Image.open(file)
        if image.width < args.min_size or image.height < args.min_size:
            continue
        del image
        dst = target_dir / file.name
        shutil.copy(file, dst)



if __name__ == '__main__':
    main()
