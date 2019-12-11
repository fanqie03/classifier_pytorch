from mtcnn import MTCNN
import argparse
import cv2
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['images', 'camera', 'videos', 'video'], default='images')
parser.add_argument('--input_path', default=0)
parser.add_argument('--output_path')
parser.add_argument('--steps_threshold', default=[0.6, 0.7, 0.7], nargs='+')

args = parser.parse_args()

# RGB format
detector = MTCNN(steps_threshold=args.steps_threshold)


def detect_and_save(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        faces = detector.detect_faces(input_img)
        for i, face in enumerate(faces):
            # if face.get('confidence') > 0.95:
            #     continue
            x, y, width, height = face.get('box')
            x2, y2 = x + width, y + height
            crop_img = img[y:y2, x:x2, :]
            cv2.imshow('crop', crop_img)
            cv2.waitKey(1)

            if args.output_path:
                filepath = os.path.join(args.output_path, f'{time.time()}.jpg')
                cv2.imwrite(filepath, crop_img)
            # 只要第一个人脸
            break
    except Exception as e:
        print(e)


def read_frame_from_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('raw', frame)
        cv2.waitKey(1)
        yield frame


if args.type == 'images':
    root = Path(args.input_path)
    images_path = list(root.rglob('*.jpg'))
    for image_path in images_path:
        img = cv2.imread(str(image_path))
        detect_and_save(img)

elif args.type == 'camera':
    cap = cv2.VideoCapture(args.input_path)
    for img in read_frame_from_video(cap):
        detect_and_save(img)
    cap.release()

elif args.type == 'videos':
    root = Path(args.input_path)
    videos_path = list(root.glob('*.mp4')) + \
                  list(root.glob('*.avi'))
    for video_path in videos_path:
        cap = cv2.VideoCapture(str(video_path))
        for img in read_frame_from_video(cap):
            detect_and_save(img)
        cap.release()

elif args.type == 'video':
    cap = cv2.VideoCapture(args.input_path)
    for img in read_frame_from_video(cap):
        detect_and_save(img)
    cap.release()
