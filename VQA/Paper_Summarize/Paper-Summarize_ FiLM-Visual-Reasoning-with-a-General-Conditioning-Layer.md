## FiLM: Visual Reasoning with a General Conditioning Layer

### Indexing:
- [Introduction](#Introduction)
- [Conclusion](#Conclusion)
- [References](#References)
---
### Introduction
- We introduce a general-purpose conditioning method for neural networks called FiLM: Feature-wise Linear Modulation. 
-  We show that FiLM layers are highly effective for visual reasoning — answering image-related questions which require a multi-step, high-level process —a task which has proven difﬁcult for standard deep learning methods that do not explicitly model reasoning. 

- Overview of Proposed Method

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer.png" width="600" hegiht="400" align=center/>

- Proposed FiLM layer

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_FiLM-Layer.png" width="600" hegiht="400" align=center/>

- Results on CLEVR dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Accuracy-on-CLEVR.png" width="600" hegiht="400" align=center/>

- FiLM parameters cluster

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Parameters-Visualization.png" width="600" hegiht="400" align=center/>

---
### Conclusion
- By efﬁciently manipulating a neural network’s intermediate features in a selective and meaningful manner using FiLM layers, a RNN can effectively use language to modulate a CNN to carry out diverse and multi-step reasoning tasks over an image. 

---
### References
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871.pdf)

---

