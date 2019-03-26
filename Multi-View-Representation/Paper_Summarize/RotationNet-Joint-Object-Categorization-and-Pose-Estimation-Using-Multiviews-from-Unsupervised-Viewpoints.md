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



---
### Experiments

#### Experiment on 3D model datasets


#### Experiment on a real image benchmark dataset

#### Experiment on a 3D rotated real image dataset


---
### Discussion
- We proposed RotationNet, which **jointly estimates object category and viewpoint** from each single-view image and aggregates the object class predictions obtained from a partial set of multi-view images.
- We consider that our pose estimation performance beneﬁts from **view-speciﬁc appearance information shared across classes** due to the **inter-class self-alignment**. 
- RotationNet has the limitation that each image should be **observed from one of the pre-deﬁned** viewpoints. 
---
### References
- [RotationNet](https://arxiv.org/pdf/1603.06208.pdf)

---
