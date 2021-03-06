## RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints

### Indexing:
- [Introduction](#Introduction)
- [Proposed method](#Proposed-method)
- [Experiments](#Experiments)
- [Discussion](#Discussion)
- [References](#References)

---
### Introduction

<img src="https://github.com/qiuyue1993/Notes/blob/master/Multi-View-Representation/images/Paper_Summarize-RotationNet-Framework.png" width="800" hegiht="300" align=center/>

#### Abstract
- We propose a Convolutional Neural Network (CNN)based model “RotationNet,” which takes **multi-view images** of an object as input and **jointly estimates its pose and object category**.
- Our method treats the **viewpoint labels as latent variables**, which are learned in an **unsupervised** manner during the training using an unaligned object dataset. 

**Joint estimation**
- Object **classiﬁcation and viewpoint estimation** is a tightly coupled problem, which can best **beneﬁt from their joint** estimation. 

---
### Proposed method
**Training sample**:
- M images of an object
- Category label y 
- Viewpoint variable v_i for each images in M
- **v_i are treated as latent variables that are optimized in the training process**

**Idea**
- We introduce an ``incorrect view'' class and append it to the target category class
- When v_i is correct, the possibility of correct label y_i should be 1
- When v_i is incorrect, the possibility of correct label y_i may not necessarily be high

#### Viewpoint setups for training

<img src="https://github.com/qiuyue1993/Notes/blob/master/Multi-View-Representation/images/Paper_Summarize-RotationNet-Viewpoints.png" width="450" hegiht="150" align=center/>

**Case (i): with upright orientation**
- Assume upright orientation
- Fix a specific axis as the rotation axis (e.g., the z-axis)
- Elevate by 30 degree
- 12 views for an object

**Case (ii): w/o upright orientation**
- Do not assume upright orientation
- Place virtual cameras on the M=20 vertices of a dodecahedron encompassing the object.

**Case (iii): with upright orientation and multiple elevation levels**
- Extension of case (i).
- Elevation angle varies from -90 degree to 90 degree

---
### Experiments
**Settings**:
- Baseline architecture: AlexNet
- Pre-trained on ImageNet
- Momentum SGD with a learning rate of 0.0005 and a momentum of 0.9 for optimization.

**Baseline model**
- Modified version of MVCNN
- View-pooling layer is placed after final softmax layer.
- Using average pooling instead of  max pooling.

#### Experiment on 3D model datasets
- ModelNet10 (4,899 object instances, 10 categories) and ModelNet40 (12,311 object instances, 40 categories)
- Ablation: views used for prediction; viewpoint setups; network architectures (AlexNet, VGG-M, ResNet-50)
- 

**Results**
- The method with late view-pooling is slightly better than original MVCNN with the view-pooling layer in the middle.
- The placement of viewpoints is the most important factor in multiview-based 3D object classification.

#### Experiment on a real image benchmark dataset
- RGBD dataset (300 object instances in 51 categories)
- When estimating object instance/category and pose using 12 views altogether, both accuracies are remarkably improved.

#### Experiment on a 3D rotated real image dataset
- New dataset: MIRO (Multi-view Images of Rotated Objects)
- 3D MFP to capture multi-view images
- 12 categories, 10 object instances per category

**Real-world applications**
- Viewpoint setup case (iii)
- Not only object classification but also estimation performance is improved by using multiple views.

---
### Discussion
- We proposed RotationNet, which **jointly estimates object category and viewpoint** from each single-view image and aggregates the object class predictions obtained from a partial set of multi-view images.
- We consider that our pose estimation performance beneﬁts from **view-speciﬁc appearance information shared across classes** due to the **inter-class self-alignment**. 
- RotationNet has the limitation that each image should be **observed from one of the pre-deﬁned** viewpoints. 
---
### References
- [RotationNet](https://arxiv.org/pdf/1603.06208.pdf)

---
