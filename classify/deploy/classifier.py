from ..backbone import *
import torch
import cv2
import numpy as np
import copy


class Classifier:
    def __init__(self, cfg, *args, **kwargs):
        print(cfg)
        cfg = copy.deepcopy(cfg)
        self.model = eval(cfg.model.pop('type'))(**cfg.model)
        self.device = torch.device('cuda:0')

        state_dict = torch.load(cfg.ckpt or cfg.train.resume_from)
        self.model.load_state_dict(state_dict['model'])
        self.classes = state_dict['classes']
        print(self.classes)
        self.idx_to_class = {k: v for k, v in zip(list(range(len(self.classes))), self.classes)}
        print(self.idx_to_class)

        self.model.eval()
        self.model.to(self.device)

    def predict(self, x, type='cv'):
        """

        :param x:单张图片，BGR类型
        :param type:
        :return:
        """
        x = self.preprocess(x)

        x = torch.from_numpy(x)
        x = x.to(self.device)
        with torch.no_grad():
            x = self.model(x)

        x = self.postprocess(x)
        return x

    def preprocess(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224))
        x = x.astype(np.float32)
        x = x / 255
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)

        return x

    def postprocess(self, x):
        value, idx = torch.max(x, dim=1)

        idx = idx.detach().cpu().numpy() if idx.requires_grad else idx.cpu().numpy()
        idx = idx[0]
        # idx = idx.cpu().numpy().tolist()[0]
        t = self.idx_to_class[idx]
        return t
