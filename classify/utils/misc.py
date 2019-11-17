import datetime
import json
import os
import sys
import time
import importlib

import torch
import yaml

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='PROG', add_help=False)
    group1 = parser.add_argument_group('group1', 'group1 description')
    group1.add_argument('foo', help='foo help')
    group2 = parser.add_argument_group('group2', 'group2 description')
    group2.add_argument('--bar', help='bar help')
    parser.print_help()
    print(parser.parse_args())


def merge_from_file(_dict, file: str):
    if file.endswith('.json'):
        with open(file, 'r') as f:
            _dict.update(json.load(f))
    elif file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, 'r') as f:
            _dict.update(yaml.unsafe_load(f))
    elif file.endswith('.py'):
        file = file[:-3]
        cfg = importlib.import_module(file)
        cfg = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
        _dict.update(cfg)
    return _dict


def test_merge_from_file():
    _dict = {}
    print(merge_from_file(_dict.copy(), 'demo.json'))
    print(merge_from_file(_dict.copy(), 'demo.yaml'))
    sys.path.append('..')
    print(merge_from_file(_dict.copy(), 'configs.helmet.py'))


if __name__ == '__main__':
    test_merge_from_file()


class Logger(object):
    def __init__(self, filename="log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        if message not in ('\n', '') and not message.startswith('\r'):
            message = str(time.asctime()) + ' | ' + message
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger()
    print(453453)
    print(path)
    print(type)


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = datetime.datetime.now()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = datetime.datetime.now() - self.clock[key]
        del self.clock[key]
        return interval.total_seconds()


def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, classes, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score,
        'classes': classes
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))