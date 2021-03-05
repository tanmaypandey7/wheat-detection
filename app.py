from flask import Flask
from flask import request, render_template

import os
import shutil

import config
from utils.detect_utils import detect, save_image

from waitress import serve

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/predict', methods=['POST'])
def predict():
    if request.form["selected-image"]:
        image_file = request.form["selected-image"]
        shutil.copyfile(f"input/test/{image_file}", f"{config.SOURCE}/image.jpg")
    else:
        image_file = request.form["custom-image"]
        image_location = os.path.join(
            config.UPLOAD_FOLDER,
            "image.jpg"
        )
        image_file.save(image_location)
    res = detect(config)
    save_image(res)
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


if __name__ == '__main__':
    app.run(port=80)
    # serve(app, host='0.0.0.0', port=8080)
