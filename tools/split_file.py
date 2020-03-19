import shutil
import os
import argparse
from pathlib import Path
"""
split file
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='/home/cmf/datasets/extract_data/test_predict/unknown')
    parser.add_argument('--target_dir', default='/home/cmf/datasets/extract_data/test_predict/split')
    parser.add_argument('--split_num', default=5)
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
    source_files = split(list(source_dir.rglob('*')), args.split_num)
    folder_name = source_dir.name
    for i, files in enumerate(source_files):
        dst_folder = target_dir / (folder_name + str(i))
        if not dst_folder.exists():
            os.makedirs(str(dst_folder))
        for j, file in enumerate(files):
            dst = dst_folder / file.name
            shutil.copy(file, dst)



if __name__ == '__main__':
    main()
