import numpy as np

import sys
sys.path.insert(0, "utils/")

import torch

from wbf_utils import weighted_boxes_fusion

from utils import torch_utils
from utils.utils import scale_coords, non_max_suppression
from utils.datasets import LoadImages


def run_wbf(boxes, scores, image_size=1024, iou_thr=0.4, skip_box_thr=0.34, weights=None):
    #     boxes =boxes/(image_size-1)
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    #     boxes = boxes*(image_size-1)
    #     boxes = boxes
    return boxes, scores, labels


def detect(config, save_img=False):
    weights, imgsz = config.weights, config.img_size
    source = 'static2'
    # Initialize
    device = torch_utils.select_device(config.device)
    half = False
    # Load model

    model = torch.load(weights, map_location=device)['model'].to(device).float().eval()

    dataset = LoadImages(source, img_size=256)

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
            pred = model(img, augment=config.augment)[0]
            pred = non_max_suppression(pred, config.nms_conf_thr, config.nms_iou_thr,  classes=None, agnostic=False)

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
