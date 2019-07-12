# Models with explicit reasoning module

## Indexing:

- [Neural Module Network](#Neural-Module-Network)
- [Dynamic Neural Module Network](#Dynamic-Neural-Module-Network)
- [N2NMN (End-to-end Module Network)](#N2NMN)
- [Explainable Neural Computation via Stack Neural Module Networks](#Explainable-Neural-Computation-via-Stack-Neural-Module-Networks)

- [Visual Coreference Resolution in Visual Dialog using Neural Module Networks](#Visual-Coreference-Resolution-in-Visual-Dialog-using-Neural-Module-Networks)



- [Inferring and Executing Programs for Visual Reasoning](#Inferring-and-Executing-Programs-for-Visual-Reasoning)

- [Neural-Symbolic VQA](#Neural-Symbolic-VQA)
- [The Neuro-Symbolic Concept Learner](#The-Neuro-Symbolic-Concept-Learner)

- [Compositional models for VQA Can neural module networks really count?](#Compositional-models-for-VQA-Can-neural-module-networks-really-count?)

- [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](#Learning-Conditioned-Graph-Structures-for-Interpretable-Visual-Question-Answering)

- [Relation Network](#Relation-Network)

- [References](#References)


---
## Neural Module Network
- Accept to CVPR 2016

### Introduction
#### Abstract

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Neural-Module-Network_Illustration.png" width="600" hegiht="400" align=center/>

*Intuition*
- **Shared linguistic substructure** in questions

*Proposed method*
- **Constucting** and **Learning** neural module networks

*Processes*
- **Decomposes questions** into their linguistic **substructures**
- Use those structures to **dynamically instantiate modular networks**

*Contributions*
- Describe neural module networks, a **general** architecture for discretely composing heterogeneous, jointly-trained neural modules into deep networks.


### Approach
#### Problem Definition
*Data*
- $(w, x, y):$ (natural-language question, image, answer)

*Model*
- A collection of modules ${m}$, each with associated parameters
- Network layout predictor $P$, which maps from strings to networks

#### Modules
- The modules operate on 3 basic data types: **images, unormalized attentions, labels**
- Format: TYPE\[INSTANCE\] (ARG1,...)
- Weights may be shared at both the **type and instance level**

*Find (Image->Attention)*
- Convolves every position in the input image with a **weight vector** to produce a heatmap or unnormalized attention

*Transform (Attention->Attention)*
- Multilayer perceptron with ReLUs
- Performing a **fully-connected mapping** from one attention to another
- Weights for this mapping are **distinct for each c**

*Combine (Attention\*Attention->Attention)*
- Merges two attentions into a single attention

*Describe (Image\*Attention->Label)*
- Takes an attention and the input image 
- Maps both to a distribution over labels

*Measure (Attention->Label)*
- Takes an attention alone
- Maps it to a distribution over labels

#### From strings to networks
- Two steps, parsing and composing networks

*Parsing*
- Map natural language questions to **layouts**
- Specify both the **set of modules used** to answer a given questions
- And the **connections between them**
- Using **Stanford Parser**, dependency parsing

*Layout*
- All leaves become **find modules**
- All internal nodes become **transform or combine modules**
- Root nodes become **describe or measure** modules depending on the domain


#### Answering natural language questions
- Final model combines the output from the **neural module network** with predictions from a simple **LSTM question encoder**

### Experiments
*Dataset*
- VQA dataset

*Results*

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Neural-Module-Network_Sampled-Resultspng.png" width="600" hegiht="400" align=center/>

- With overall 58.7% on VQA dataset

### Comments
- Mannual designed modules might constraint the ability of this model, Is that possible to learn the modules automatically?
- Kind like mannual designed feature extraction

---
## Dynamic Neural Module Network

- Accepted to NAACL 2016

### Introduction
*Abstract*
- Use natural language strings to **automatically assemble neural networks** from a collection of composable modules
- Parameters for these modules are learned jointly with network-assembly parameters via reinforcement learning
- The supervision is (world, question, answer) triples
- The world representations can be **images or knowledge bases**


### Approach
#### Problem Definition
*Notations*
- $w$: world representation
- $x$: question
- $y$: answer 
- $z$: network layout
- $\theta$: collection of model parameters

*Two Distributions*
- **Layout model**: choose a layout for a sentence
- **Execution model**: applies the network specified by $z$ to $w$

#### Evaluating modules
*Lookup*
- Produces attention

*Find*
- Attention
- Compute a **distributions over indices** by concatenating the parameter argument with each position of the input feature map, and passing the concatenated vector through a MLP

*Relate*
- Attention -> Attention
- Directs focus from one region of the input to another

*And*
- Attention* -> Attention
- Perform an operation analogous to set intersection for attentions

*Describe*
- Attention -> Labels
- Computes a weighted average of $w$ under the input attention

*Exists*
- Existential quantifier
- Inspects the incoming attention directly to produce a lable

### Experiments
#### Questions about images
*Dataset*
- VQA v1
- Best performance at that time

### Different from Neural Module Network
- **Learn a network structure predictor** jointly with module parameters themselves
- Extend visual primitives from previous work to reason over **structured world representations**

### Comments
- Suited for open-domain VQA baseline experiments
- How many instances does a module have? 

---
## N2NMN

- Accepted to ICCV 2017

### Introduction

- Illustration

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_N2NMN_Illustration.png" width="600" hegiht="400" align=center/>

*Problems of NMN*
- Rely on brittle **off-the-shelf** parsers
- Restricted to the module configurations proposed by these parsers

*Proposal*
- Learn to reason by **directly predicting instance-specific network layouts** (Sequence-to-sequence RNN)
- Learn to generate **network structures** and **network parameters**

*Contributions*
- A method for learning a **layout policy that dynamically predicts a network structure for each instance**
- A **module parameterization** that uses a **soft attention over question words** rather than hard-coded word assignments

*Comparing with Prior Works*
- Learn to **optimize over the full space of network layouts**
- Requires **no parser**

### Approach

- Model Overview

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_N2NMN_Model-Overview.png" width="600" hegiht="400" align=center/>

*Two main components*
- A set of **co-attentive neural modules** that provide **parameterized functions for solving sub-tasks**
- A **layout policy** to predict a **question-specific layout** 

#### Attentional neural modules

- Neural Module List

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_N2NMN_Module-List.png" width="600" hegiht="400" align=center/>

*Compositions*
- A set of **neural modules**
- A module $m$ is parameterized function $y = f_m (a_1, a_2, ...; x_{vis}, x_{txt}, \theta_m)$ 
- Input: $a_1, a_2, ...$, image attention map over the convolutional image feature grid
- Additional Information: $x_{vis}$ indicates image features; $x_{txt}$ indicates question features
- Output: $y$ is either an image attention map or a probability distribution

- **Obtains the textual input using soft attention over question words**

#### Layout policy with sequence-to-sequence RNN
- Input: question $q$
- Output: probability distribution

*Possible output layouts*
- The layout policy **predicts a distribution over the space of all possible layouts**
- Every possible layout can be represented as a **syntax tree**, can be mapped into a linearized sequence using **Reverse Polish Notation**

*Layout prediction process*
- **Sequence-to-sequence** learning problem from **questions to module tokens**
- Attentional Recurrent Neural Network

*Test time*
- predict a maximum-probability layout using beam search

#### End-to-end training
- Loss is not fully differentiable since the layout is discrete
- Optimize network using **backpropagation for differentiable parts**
- **Policy gradient** in reinforcement learning for **non-differentiable part**
- Introduced a **simple baseline implemented as an exponential moving average over recent loss**

*Behavioral cloning from expert polices*
- Treat **candidate layouts as an existing expert policy** to provide additional supervision
- Pretrain model by **behavioral cloning** by minimizing the **KL-divergence** between the expert policy and layout policy

### Experiments
#### SHAPES dataset
- 15,616 image-question pairs with 244 unique questions
- Each image consists of shapes of different colors and sizes

*Two settings*
- Behavioral cloning from expert: 100% Accuracy on SHAPES
- Policy search from scratch: 96.19% Accuracy on SHAPES
- NMN: 90.80%

#### CLEVR dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_N2NMN_CLEVR-Results.png" width="600" hegiht="400" align=center/>

- Focuses mostly on the reasoning ability
- Feature extraction with VGG and added extra location information
- 83.7% overall accuracy on CLEVR 
- **Soft attention module parameterization is better than the hard-coded textual parameters in NMN**
- **The performance consistently improves after end-to-end training with policy search using reinforcement learning**

#### VQA dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_N2NMN_VQAv1-Results.png" width="600" hegiht="400" align=center/>

- Construct an expert layout policy using a **syntactic parse of questions**
- Better than MCB on VQA v1
### Comments
- Why introduced a **simple baseline** and how can it be used?
- The model with cloning expert performs far more well than RL from scratch
---
## Explainable Neural Computation via Stack Neural Module Networks

- Accept to ECCV 2018




---
## Visual Coreference Resolution


---
## Inferring and Executing Proagrams for Visual Reasoning

---
## Neural-Symbolic VQA


---
## The Neuro-Symbolic Concept Learner





--
## Compositional models for VQA Can neural module networks really count?


--
## Learning Conditioned Graph Structures for Interpretable Visual Question Answering

---
## Relation Network

---
## References
- [Neural Module Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf)
- [Dynamic Neural Module Networks](https://arxiv.org/pdf/1601.01705.pdf)
- [N2NMN (End-to-end Module Network)](https://arxiv.org/pdf/1704.05526.pdf)
- [Explainable Neural Computation via Stack Neural Module Networks](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Ronghang_Hu_Explainable_Neural_Computation_ECCV_2018_paper.pdf)

- []
- [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](https://arxiv.org/pdf/1806.07243.pdf)
---
