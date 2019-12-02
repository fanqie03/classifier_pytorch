import cv2
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('output')
opt = parser.parse_args()

cap = cv2.VideoCapture(opt.video)
ret, frame = cap.read()
while ret:
    filename = str(time.time()) + '.jpg'

    path = os.path.join(opt.output, filename)

    cv2.imwrite(path, frame)

    ret, frame = cap.read()
