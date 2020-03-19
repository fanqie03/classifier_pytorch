import cv2
import numpy as np


def draw_pane(img, bbox, pane_info, alpha=0.5):
    """
    在img旁边画一个半透明的矩形框,填充pane_info数组中的内容
    :param img:
    :param bbox: corner format
    :param pane_info:
    :param alpha:
    :return:
    """
    # 检查和纠正参数
    if not isinstance(pane_info, list):
        pane_info = [pane_info]

    pane_info = [str(x) for x in pane_info]
    if isinstance(bbox, list):
        bbox = [int(x) for x in bbox]
    if isinstance(bbox, np.ndarray):
        bbox = bbox.astype(np.int)

    fontFace, fontScale, thickness = 1, 1, 1
    width = bbox[2] - bbox[0]
    pane_info_length = [len(x) for x in pane_info]
    long_index = np.argmax(pane_info_length)
    img_height, img_width, img_channel = img.shape
    text_width, text_height = cv2.getTextSize(pane_info[long_index], fontFace, fontScale, thickness)[0]
    pane_width, pane_height = text_width, text_height * len(pane_info)
    pane_bbox = [bbox[0] + width, bbox[1], bbox[0] + width + pane_width, bbox[1] + pane_height]

    pane_bbox[2] = np.clip(pane_bbox[2], 0, img_width - 1)
    pane_bbox[3] = np.clip(pane_bbox[3], 0, img_height - 1)
    pane_width, pane_height = pane_bbox[2] - pane_bbox[0],pane_bbox[3] - pane_bbox[1]

    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255, 255, 255))
    roi = img[pane_bbox[1]: pane_bbox[3], pane_bbox[0]: pane_bbox[2]]
    pane = np.ones((pane_height, pane_width, img_channel), np.uint8) * 125
    for i, info in enumerate(pane_info):
        cv2.putText(pane, info, (0, text_height * (i + 1)), fontFace=fontFace, fontScale=fontScale,
                    color=(255, 0, 0), thickness=thickness)
    img[pane_bbox[1]: pane_bbox[3], pane_bbox[0]: pane_bbox[2]] = cv2.addWeighted(roi, alpha, pane, 1 - alpha, 0)


img = cv2.imread('0.jpg')

bbox = [120, 130, 220, 230]

draw_pane(img, bbox, ['1', '2.0', '33.0'])

cv2.imshow('', img)
cv2.waitKey()
