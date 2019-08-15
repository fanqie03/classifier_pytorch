import os
import random
from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser(description='extract subset dataset')
parser.add_argument('--input_folder', default='/home/cmf/datasets/extract_data/gongdi/person')
parser.add_argument('--output_folder', default='/home/cmf/datasets/extract_data/gongdi_subset/person')
parser.add_argument('--random_seed', default=None)
parser.add_argument('--subset_percentage', default=0.03)
parser.add_argument('--k', default=400)
args = parser.parse_args()

print(args)

input_folder = Path(args.input_folder)
output_folder = Path(args.output_folder)
random.seed(args.random_seed)


if not output_folder.exists():
    os.makedirs(str(output_folder))

# catagorical_folder = input_folder.rglob('*')
#
# for catagorical in catagorical_folder:
#     output_catagorical = output_folder / catagorical.name
#     if not output_catagorical.exists():
#         output_catagorical.mkdir()
#     files = list(catagorical.rglob('*'))
#     print(len(files))
#     sub_num = int(len(files) * args.subset_percentage)
#     sub_files = random.sample(files, k=sub_num)
#     print(len(sub_files))
#     for file in sub_files:
#         dst = output_catagorical / file.name
#         shutil.copy(str(file), str(dst))

src = list(input_folder.rglob('*'))
sub_src = random.sample(src, k=args.k)
for i in sub_src:
    dst = output_folder / i.name
    shutil.copy(str(i), str(dst))