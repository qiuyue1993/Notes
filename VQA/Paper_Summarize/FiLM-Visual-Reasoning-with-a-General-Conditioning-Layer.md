## FiLM: Visual Reasoning with a General Conditioning Layer

### Indexing:
- [Introduction](#Introduction)
- [Method](#Method)
- [Related Work](#Related-Work)
- [Conclusion](#Conclusion)
- [References](#References)
---
### Introduction


- We introduce a **general-purpose** conditioning method for neural networks called FiLM: **Feature-wise Linear Modulation**. 
- FiLM layers influence neural network computation via a simple, feature-wise affine transformation based on conditioning information. 
-  We show that FiLM layers are **highly effective for visual reasoning**. 

- In the case of visual reasoning, FiLM layers **enable a Recurrent Neural Network (RNN) over an input question to influence Convolutional Neural Network (CNN) computation** over an image.



- Results on CLEVR dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Accuracy-on-CLEVR.png" width="600" hegiht="400" align=center/>

- FiLM parameters cluster

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Parameters-Visualization.png" width="600" hegiht="400" align=center/>

---
### Method
#### Feature-wise Linear Modulation
FiLM learns to adaptively **influence the output of a neural network** by applying an **affine transformation**, or FiLM, to the network’s intermediate features, **based on some input**.

- Proposed FiLM layer

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_FiLM-Layer.png" width="150" hegiht="100" align=center/>

**FiLM**:
$\gamma_{i,c} = f_c (x_i), \qquad   \beta_{i,c} = h_c (x_i)$
- input: $x_i$
- learns functions $f$ and $h$ which output,
- $\gamma_{i,c}$ and $\beta_{i,c}$
- $f$ and $h$ can be arbitrary functions such as neural networks.
- It is often beneficial to share parameters across $f$ and $h$


#### Model

- Overview of Proposed Method

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer.png" width="300" hegiht="200" align=center/>



---
### Conclusion
- By efficiently manipulating a neural network’s intermediate features in a selective and meaningful manner using FiLM layers, a **RNN can effectively use language to modulate a CNN** to carry out diverse and **multi-step reasoning tasks** over an image. 
- Notably, we provide evidence that **FiLM’s success is not closely connected with normalization** as previously assumed. Thus, we open the door for applications of this approach to settings where normalization is less common, such as RNNs and reinforcement learning. 

---
### References
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871.pdf)

---

