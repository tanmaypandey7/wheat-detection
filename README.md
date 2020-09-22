# Wheat Detection
Detecting wheat heads using YOLOv5
- [Web App demo](#Web-app-demo)
- [Brief overview of the competition images](#Brief-overview-of-the-competition-images)
- [Modifications](#Modifications)
- [Training](#Training)
- [Inference and Deployment](#Inference-and-Deployment)

## Web App Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bZe1CDa4g7wnOUZFO9ZUmw15T2I6Po0w?usp=sharing)

[![https://imgur.com/a/Ap2kaeX](http://img.youtube.com/vi/JrL8nsV53tc/0.jpg)](http://www.youtube.com/watch?v=JrL8nsV53tc "Wheat App")

## Brief overview of the competition images
Wheat heads were from various sources:  
<a href="https://imgur.com/HhOQtba"><img src="https://imgur.com/HhOQtba.jpg" title="head" alt="head" /></a>  
A few labeled images are as shown: (Blue bounding boxes)  
<a href="https://imgur.com/QhnuEEf"><img src="https://imgur.com/QhnuEEf.jpg" title="head" alt="head" width="378" height="378" /></a> <a href="https://imgur.com/5yUJCPV"><img src="https://imgur.com/5yUJCPV.jpg" title="head" alt="head" width="378" height="378" /></a>  

## Pre-trained models
Models can be downloaded from <a href="https://www.kaggle.com/ii5m0k3ii/mixup50e">here</a>. (Use last_yolov5x_4M50fold0.pt for best results) 

## Modifications
The YOLOv5 notebook internally does some augmentations while preparing a Dataset. 
Originally, this Dataset consists of only Mosaic images.

**Mosaic** - https://arxiv.org/pdf/2004.12432.pdf  
      4 images are cropped and stitched together  
    <a href="https://imgur.com/YZn47iN"><img src="https://imgur.com/YZn47iN.jpg" title="head" alt="head" width="378" height="378" /></a>
    
Here, I modified the repo to add Mixup.

**Mixup** - https://arxiv.org/pdf/1710.09412.pdf  
      2 images are mixed together
    <a href="https://imgur.com/HkDFQ2g"><img src="https://imgur.com/HkDFQ2g.jpg" title="head" alt="head" /></a>
 
I modified the code(especifically utils.datasets) so it had a 50-50 chance of creating a mixup or a mosaic image. This was very helpful for us as it boosted our public score from 0.77->0.7769. 

These developments were made before we found out that YOLOv5 was non-compliant and had to switch to EfficientDet for our final 2 submissions.
Kaggle later updated the leaderboard with the final 2 submissions and we ended up at 113th Private(Top 6%).

## Training
We trained the model for 50 epochs on Colab Pro. 

## Inference and Deployment
Our best model is currently being used for inference in this web-app. I uses HTML and CSS as front-end and Flask as the backend.
This web-app is served on Google Colab but can be easily deployed on AWS or GCP as well.
