import cv2
import argparse
import torch
from classify.backbone import *
from classify.config import get_default_args
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('--input_video')
# parser.add_argument('--output_video')
# parser.add_argument('--config')
# # parser.add_argument('--model')
# # parser.add_argument('--num_classes')
# parser.add_argument('--ckpt')
# parser.add_argument('--show', default=False, action='store_true')

args = get_default_args()

cap = cv2.VideoCapture(args.input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

paths = args.input_video.split('.')

output_video = args.output_video or paths[0] + '_.' + paths[1]
print(output_video)
out = cv2.VideoWriter(output_video, fourcc, fps, size)

ret, frame = cap.read()

device = torch.device('cuda')

m = torch.load(args.ckpt)
print(m.keys())
classes = m['classes']
net = eval(args.model.type)(args.model.num_classes)
net.load_state_dict(m['model'])
net.to(torch.device('cuda'))
net.eval()

while ret:
    raw = frame
    # 模型处理

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    # img = np.float32(img)
    img = img.astype(np.float32)
    # print(img.max(), img.min())
    img = img / 255
    # print(img.max(), img.min())
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)
    img = img.cuda()

    # img = torch.rand_like(img)
    print(img.max(), img.min())
    with torch.no_grad():
        r = net(img)
        value, index = torch.max(r, 1)
    t = classes[index[0]]
    print(r, t)
    ...
    if t == 'no':
        cv2.putText(raw, t, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0))
    elif t == 'yes':
        cv2.putText(raw, t, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255))
    else:
        cv2.putText(raw, t, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))
    draw = raw
    # if args.show:
#     cv2.imshow('draw', draw)
    cv2.imshow('raw', raw)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # # 写入一帧
    out.write(draw)
    # # 读取新的一帧
    ret, frame = cap.read()

# 关闭
out.release()
cap.release()
