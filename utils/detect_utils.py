import numpy as np

import sys
sys.path.insert(0, "utils/")

import torch

from wbf_utils import weighted_boxes_fusion

from utils import torch_utils
from utils.utils import scale_coords, non_max_suppression
from utils.datasets import LoadImages

import cv2
from PIL import Image

def run_wbf(boxes, scores, image_size=1024, iou_thr=0.4, skip_box_thr=0.34, weights=None):
    #     boxes =boxes/(image_size-1)
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    #     boxes = boxes*(image_size-1)
    #     boxes = boxes
    return boxes, scores, labels


def detect(config, save_img=False):
    weights, imgsz = config.WEIGHTS, config.IMG_SIZE
    source = config.SOURCE
    # Initialize
    device = torch_utils.select_device(config.DEVICE)
    half = False
    # Load model

    model = torch.load(weights, map_location=device)['model'].to(device).float().eval()

    dataset = LoadImages(source, img_size=config.IMG_SIZE)

    all_path = []
    all_bboxex = []
    all_score = []
    for path, img, im0s, vid_cap in dataset:
        print(im0s.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        bboxes_2 = []
        score_2 = []
        if True:
            pred = model(img, augment=config.AUGMENT)[0]
            pred = non_max_suppression(pred, config.NMS_CONF_THR, config.NMS_IOU_THR, classes=config.CLASSES, agnostic=config.AGNOSTIC_NMS)

            bboxes = []
            score = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    for *xyxy, conf, cls in det:
                        if True:  # Write to file
                            xywh = torch.tensor(xyxy).view(-1).numpy()  # normalized xywh
                            bboxes.append(xywh)
                            score.append(conf)
            bboxes_2.append(bboxes)
            score_2.append(score)
        all_path.append(path)
        all_score.append(score_2)
        all_bboxex.append(bboxes_2)
    return all_path, all_score, all_bboxex


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def save_image(res):
    all_path, all_score, all_bboxex = res
    idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(all_path[idx], cv2.IMREAD_COLOR)
    # fontScale
    fontScale = 1
    boxes = all_bboxex[idx]
    scores = all_score[idx]
    # Blue color in BGR
    color = (255, 0, 0)

    for row in range(len(all_path)):
        boxes = all_bboxex[row]
        scores = all_score[row]
        boxes, scores, labels = run_wbf(boxes, scores)
        boxes = (boxes * 1024 / 1024).astype(np.int32).clip(min=0, max=1023)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    # Line thickness of 2 px
    thickness = 2
    for b, s in zip(boxes, scores):
        image = cv2.rectangle(image, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 0, 0), 3)
        image = cv2.putText(image, '{:.2}'.format(s), (b[0], b[1]), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    im = Image.fromarray(image[:, :, ::-1])
    im.save("static/pred_image.jpg")


def run_wbf(boxes, scores, iou_thr=0.5, skip_box_thr=0.43):
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    return boxes, scores, labels

