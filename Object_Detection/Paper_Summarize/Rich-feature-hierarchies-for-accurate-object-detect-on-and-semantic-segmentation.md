## Rich feature hierarchies for accurate object detection and semantic segmentation

### Indexing
- [Introduction](#Introduction)
- [References](#References)

---
### Introduction

<img src="https://github.com/qiuyue1993/Notes/blob/master/Object_Detection/Images/Paper-Summarize_R-CNN_Overall-framework.png" width="600" hegiht="400" align=center/>

**Overall Process**
- Input image
- Extract region proposals (~2k) (selective search)
- Compute CNN features
- Classify regions
- Bounding boxes Regression

**Training Process**
- Pretrain CNN on ImageNet.
- Fine-tuning CNN on PASCAL VOC.

**Bounding box regression**
- 2000\*20 vector (2000:seletcted bounding boxes; 20: class number)
- For every column of the vector (20 column indicates 20 classes), rank by the value
- Choose the bounding box with the highest score, compute the IoU with the remains, if the IoU > threshold, remove the bbox with low IoU
score otherwise save both of the bounding boxes.
- Iterate the above operation till the last bbox.

**Deficiency of R-CNN**
- Repeated calculation (run CNN for every ROI with a total number of 2,000)
- Multiple-stage pipeline (region proposals, ConvNet, SVM, BBox Regression)
- High time and spatial consumption

---
### References
- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
- [Summarize Reference](https://zhuanlan.zhihu.com/p/38946391)
