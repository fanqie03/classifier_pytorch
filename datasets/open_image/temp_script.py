import pandas as pd
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_csv',
                        default='/home/cmf/datasets/open-image/annotations_origin/train-annotations-bbox.csv')
    parser.add_argument('--target_csv',
                        default='/home/cmf/datasets/open-image/annotations/train-annotations-bbox.csv')
    parser.add_argument('--image_folder', default='/home/cmf/datasets/open-image/train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    p = Path(args.image_folder)
    files = [x.name.split('.')[0] for x in p.rglob('*')]
    df = pd.read_csv(args.origin_csv)
    df_sub = df[df['ImageID'].isin(files)]
    df_sub.to_csv(args.target_csv, index=False)
