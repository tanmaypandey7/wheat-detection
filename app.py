from flask import Flask
from flask import request
from flask import render_template
from utils.detect_utils import detect
import os
import sys

sys.path.append("./src")

import cv2
import matplotlib.pyplot as plt
from utils.wbf_utils import weighted_boxes_fusion
from PIL import Image
import utils
import config

import numpy as np

app = Flask(__name__)
DEVICE = "cpu"
MODEL = None
UPLOAD_FOLDER = "static"


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files["image"]
    image_location = os.path.join(
        UPLOAD_FOLDER,
        "image.jpg"
    )
    # image_file.save(image_location)
    image_file.save(image_location)
    # config.img_size = image_file
    res = detect(config)
    # pred = detect(config)
    # pred = 1
    all_path, all_score, all_bboxex = res
    # pred_image[1].save(image_location)
    size = 300
    idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = image = cv2.imread(all_path[idx], cv2.IMREAD_COLOR)
    # fontScale
    fontScale = 1
    boxes = all_bboxex[idx]
    scores = all_score[idx]
    # Blue color in BGR
    color = (255, 0, 0)

    for row in range(len(all_path)):
        image_id = all_path[row].split("/")[-1].split(".")[0]
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
        image = cv2.putText(image, '{:.2}'.format(s), (b[0] + np.random.randint(20), b[1]), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    im = Image.fromarray(image[:, :, ::-1])
    im.save("static/pred_image.jpg")
    return render_template("index.html", image_loc='pred_image.jpg')


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('index.html')


def run_wbf(boxes, scores, iou_thr=0.5, skip_box_thr=0.43):
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    return boxes, scores, labels


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
