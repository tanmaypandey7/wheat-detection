# Wheat Detection
Detecting wheat heads using YOLOv5

- [Brief overview of the competition images](#Brief-overview-of-the-competition-images)
- [Modifications](#Modifications)
- [Training](#Training)
- [Inference and Deployment](#Inference and Deployment)

## Brief overview of the competition images
Wheat heads were from various sources:  
<a href="https://imgur.com/HhOQtba"><img src="https://imgur.com/HhOQtba.jpg" title="head" alt="head" /></a>  
A few labeled images are as shown: (Blue bounding boxes)  
<a href="https://imgur.com/QhnuEEf"><img src="https://imgur.com/QhnuEEf.jpg" title="head" alt="head" width="378" height="378" /></a> <a href="https://imgur.com/5yUJCPV"><img src="https://imgur.com/5yUJCPV.jpg" title="head" alt="head" width="378" height="378" /></a>  

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
 
I modified the code(especifically utils.datasets) so it had a 50-50 chance of creating a mixup or a mosaic image. This was very helpful for us as it boosted our score from 0.77->0.7769. 
    
## Training
We trained the model for 50 epochs on Colab Pro. 

## Inference and Deployment
Our best model is currently being used for inference in this web-app. I uses HTML and CSS as front-end and Flask as the backend.
This web-app is served on t2.medium EC2 instance.
