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
Overall model contains:
- FiLM-generating linguistic pipeline.
- FiLM-ed visual pipeline.

##### FiLM generator
- Process a question $x_i$ using GRU 
- GRU Input: 200-dim word embeddings
- GRU hidden units: 4, 096
- Predict $(\gamma_{i, \cdot}^n, \beta_{i, \cdot}^n)$ for each $n^th$ residual block via affine projection

##### Visual Pipeline
- Input: $224\times224$ image
- Output: 128 $14\times14$ image feature maps
- Using either a CNN trained from scratch or fixed, pre-trained feature extractor

- Overview of Proposed Method

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer.png" width="300" hegiht="200" align=center/>


---
### Related Work
- Conditional Normalization (CN): We seek to show that feature-wise affine conditioning is effective for multi-step reasoning and understand the underlying mechanism behind its success.
- Concatenate constant feature maps of conditioning information with convolutional layer input.
- Other methods gate an input's features as a function of that same input, rather than a separate conditioning input.

- Visual Reasoning: Program Generator + Execution Engine model.
- Visual Reasoning: Relation Networks (RNs)

---
### Experiments


#### CLEVR Task

##### Baselines
- Q-type baseline
- LSTM
- CNN+LSTM
- Stacked Attention Networks (CNN + LSTM + SA)
- End-to-End Module Networks (N2NMN) and Program Generator + Execution Engine (PG+EE)
- Relation Networks (CNN + LSTM + RN)

##### Results
- Results on CLEVR dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Accuracy-on-CLEVR.png" width="600" hegiht="400" align=center/>


- FiLM achieves a new overall state-of-the-art on CLEVR.
- FiLM performs equally well using raw pixel inputs comparing with pre-trained image features.

#### What Do FiLM Layers Learn?

##### Activation Visualizations

- Activation Visualization Examples 

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Activation-visualization.png" width="600" hegiht="200" align=center/>

- These images reveal that the FiLM model predicts using features of areas near answer-related or question-related objects, as the high CLEVR accuracy also suggests. 

-  Final **MLP itself carries out some reasoning**, using FiLM to extract relevant features for its reasoning. 


##### FiLM Parameter Histograms
- $\gamma$ and $\beta$ values take advantage of a sizable range.
- $\gamma$ values show a sharp peak at 0, showing that FiLM learns to use the question to shut off or significantly supress whole feature maps.
- $\gamma$ and $\beta$ also are used to be selective about which activations pass the ReLU.

##### FiLM Parameters t-SNE Plot
- FiLM parameters cluster

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarize_FiLM-Visual-Reasoning-with-a-General-Conditioning-Layer_Parameters-Visualization.png" width="600" hegiht="400" align=center/>

-  FiLM layers learn a sort of function-based modularity without an architectural prior. 


#### Ablations

##### Effect of $\gamma$ and $\beta$
- FiLM can learn to condition the CNN for visual reasoning through either biasing or scaling alone, albeit not as well as conditioning both together. 
- $\gamma$ is more important than $\beta$
- Noise in $\gamma$ hurts performance signiﬁcantly more,  showing FiLM’s higher sensitivity to changes in $\gamma$ than in $\beta$ and corroborating the relatively greater importance of $\gamma$. 

##### Restricting $\gamma$
- Restrictions hurt performance.
-  FiLM’s ability to scale features by large magnitudes appears to contribute to its success. 
-  FiLM’s capacity to negate and zero out feature maps is important.

##### Conditional Normalization
- No substantial performance drop when moving FiLM layers to different parts of our model’s ResBlocks; 

##### Repetitive Conditioning
- The model can reason and answer diverse questions successfully by modulating features even just once

##### **Spatial Reasoning**
-  FiLM models are able to reason about space simply from the spatial information contained in a single location of ﬁxed image features. 

##### Residual Connection
- **Removing the residual connection causes one of the larger accuracy drops.** 
- The best model learns to primarily use features of locations that are repeatedly important throughout lower and higher levels of reasoning to make its ﬁnal decision. 

##### Model Depth
- FiLM is robust to varying depth.

#### CLEVR-Humans: Human-Posed Questions

##### Method
 - To test FiLM on CLEVR-Humans, we take our best CLEVR-trained FiLM model and ﬁne-tune its FiLMgenerating linguistic pipeline alone on CLEVR-Humans. 

##### Results
- FiLM is well-suited to handle more complex and diverse questions.
- These results thus provide some evidence for the beneﬁts of FiLM’s **general nature**.


#### CLEVR Compositional Generalization Test

##### Results
- FiLM surpasses other visual reasoning models at learning general concepts. 

##### Sample Efficiency and Catastrophic Forgetting
- FiLM achieves prior state-of-the-art accuracy with 1/3 as much ﬁne-tuning data. 
- However, our FiLM model still suffers from catastrophic forgetting after ﬁne-tuning. 

##### Zero-Shot Generalization
- Our method simply allows FiLM to take advantage of any concept disentanglement in the CNN after training. 
- **However, approaches from word embeddings, representation learning, and zero-shot learning can be applied to directly optimize $\gamma$ and $\beta$ for analogy-making** 

---
### Conclusion
- By efficiently manipulating a neural network’s intermediate features in a selective and meaningful manner using FiLM layers, a **RNN can effectively use language to modulate a CNN** to carry out diverse and **multi-step reasoning tasks** over an image. 
- Notably, we provide evidence that **FiLM’s success is not closely connected with normalization** as previously assumed. Thus, we open the door for applications of this approach to settings where normalization is less common, such as RNNs and reinforcement learning. 

---
### References
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871.pdf)

---

