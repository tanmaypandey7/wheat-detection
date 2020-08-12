from flask import Flask
from flask import request
from flask import render_template
from utils.detect_utils import detect
import os
import sys
sys.path.append("./src")
from flask import request
import shutil
import cv2
from utils.wbf_utils import weighted_boxes_fusion
from PIL import Image
import config

import numpy as np

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
DEVICE = "cpu"
MODEL = None
UPLOAD_FOLDER = "static2"


@app.route('/predict', methods=['POST'])
def predict():
    if request.form["selected-image"]:
        image_file = request.form["selected-image"]
        shutil.copyfile(f"images/{image_file}", "static2/image.jpg")
    else:
        image_file = request.form["custom-image"]
        image_location = os.path.join(
            UPLOAD_FOLDER,
            "image.jpg"
        )
        image_file.save(image_location)
    res = detect(config)
    all_path, all_score, all_bboxex = res
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
    return render_template("index.html", image_loc="static/pred_image.jpg")

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

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
