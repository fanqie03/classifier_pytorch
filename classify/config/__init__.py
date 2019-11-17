from easydict import EasyDict as edict
import argparse
from classify import merge_from_file

def get_default_args():
    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--config', help='train config file path', default=['classify/config/defaults.yaml', 'configs.test.py'], nargs='+')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    args = parser.parse_args()

    args = edict(args.__dict__)

    # if args.config:
    for cfg in args.config:
        merge_from_file(args, cfg)

    return args
