# Detection-of-Multiple-Defects-on-Bridge-Surfaces-with-Improved-YOLOv8

This repo contains the official **PyTorch** code for YOLOv8-CBAM-Wise-IoU .

## Introduction

<p align="center">
    <img src="figures/YOLOv8CBAM结构图新.png" width= "600">
</p>

### Key Features:
(1)The innovative addition of the CBAM module in YOLOv8 is an attention mechanism that adaptively adjusts channels and spatial weights in feature maps to improve the generalization and perception performance of defect detection models. 
(2)The innovative use of the Wise-IoU loss function in YOLOv8, which has a gradient gain recognition allocation strategy, strategically reduces the competitiveness of high quality anchor boxes and the adverse effects of low quality anchor boxes on gradients. This enables the model to prioritize average-quality anchor boxes, further enhancing its detection accuracy. 
(3)This paper proposes YOLOv8-CBAM-Wise-IoU model by integrating the advantages of CBAM module and Wise-IoU loss function, and validates the performance of the proposed model through the task of multi-defect detection on bridge surfaces. 


### Method 

#### The CBAM module

<p align="center">
    <img src="figures/Fig2.jpg" width= "600">
</p>

RFA can be considered a lightweight, plug-and-play module, with its structure being a fixed convolutional com-bination. RFA relies on the assistance of convolution operations, while convolution operations also benefit from RFA to enhance performance.

<br>

#### Context Broadcasting Median

<p align="center">
    <img src="figures/Fig5.jpg" width= "600">
</p>
We design and proposes the Context Broadcasting Median (CBM) module, specifically for metal surface defect de-tection. Extensive ablation experiments demonstrate the superior performance of this module in metal surface defect detection tasks.

<br>

### High accuracy and fast convergence

<br>

<p align="center">
    <img src="figures/Fig6(a).jpg" width="400" style="margin-right: 10px;">
    <img src="figures/Fig6(b).jpg" width="400" style="margin-left: 10px;">
</p>
RFAConv Module Enhances Model Convergence Speed. (a) The training loss decreases faster with the RFAConv module. (b) The validation loss is lower with the RFAConv module.

<br>

<p align="center">
    <img src="figures/Fig7(a).png" width="400" style="margin-right: 10px;">
    <img src="figures/Fig7(b).png" width="400" style="margin-left: 10px;">
</p>
ROC Curves Before and After Adding the RFAConv Module. (a) ViT-B model ROC curves. (b) ViT-B with RFAConv module ROC curves.

<br>

## Dependencies

- Python 3.8
- PyTorch == 1.13.0
- torchvision == 0.12.0
- fvcore == 0.1.5
- numpy
- timm == 0.4.12
- yacs


## Dataset

* aluminum surface defect dataset:

original：https://tianchi.aliyun.com/dataset/140666<br>
Used in the paper：[https://www.kaggle.com/datasets/wehaoreal/aluminum-profile-surface-defects-data-set](https://www.kaggle.com/datasets/weihaoreal/aluminum-profile-surface-defects-data-set)

* X-SSD hot-rolled steel strip:

[https://www.kaggle.com/datasets/sayelabualigah/x-sdd](https://www.kaggle.com/datasets/sayelabualigah/x-sdd)

* nut:

[https://uverse.rboflow.com/rocz/nutsqnfzt/dataset/2](https://www.kaggle.com/datasets/weihaoreal/nut-surface-defect-dataset)
