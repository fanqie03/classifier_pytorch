import requests
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed, ProcessPoolExecutor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('output_dir')
    return parser.parse_args()


def download_image(dir, url, name=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(url)
    file_name = name or url.split('/')[-1]
    file_path = os.path.join(dir, file_name)
    if os.path.exists(file_path):
        return

    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)


def main():
    args = get_args()
    p = Path(args.output_dir)
    executor = ProcessPoolExecutor()
    all_task = []
    df = pd.read_csv(args.csv_file)

    if df.get('wubao') is not None:
        df = df[df['wubao']!=1]

    unique_df = df.drop_duplicates('face_url', 'first')

    for name, face_url in tqdm(zip(unique_df['name'], unique_df['face_url'])):
        sub_dir = str(p / name)
        all_task.append(executor.submit(download_image, sub_dir, face_url, name + '.jpg'))
        # download_image(sub_dir, face_url)

    # for name, image_url in tqdm(zip(df['name'], df['image_url'])):
    #     # print(name, face_url, image_url)
    #     sub_dir = str(p / name)
    #     # executor.map(download_image, (str(sub_dir), face_url))
    #     # executor.map(download_image, (str(sub_dir), image_url))
    #     all_task.append(executor.submit(download_image, sub_dir, image_url))
    #     # download_image(sub_dir, image_url)


    import time
    time.sleep(10)
    # for task in tqdm(as_completed(all_task)):
    #     data = task.result


if __name__ == '__main__':
    main()
