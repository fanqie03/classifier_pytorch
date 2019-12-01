import cv2
import argparse
import torch
from classify.backbone import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_video')
parser.add_argument('--output_video')
parser.add_argument('--config')
parser.add_argument('--model')
parser.add_argument('--num_classes')
parser.add_argument('--ckpt')
parser.add_argument('--show', default=False, action='store_true')

opt = parser.parse_args()

cap = cv2.VideoCapture(opt.input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(opt.output_video, fourcc, fps, size)

ret, frame = cap.read()

while ret:
    raw = frame
    # 模型处理
    net = eval(opt.model)(opt.num_classes)
    net.load_state_dict(torch.load(opt.ckpt))
    net.to(torch.device('cuda'))

    ...
    draw = raw
    if opt.show:
        cv2.imshow('raw', raw)
        cv2.imshow('draw', draw)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # 写入一帧
    out.write(draw)
    # 读取新的一帧
    ret, frame = cap.read()

# 关闭
out.release()
cap.release()
