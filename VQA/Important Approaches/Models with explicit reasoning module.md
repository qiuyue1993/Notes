# Models with explicit reasoning module

## Indexing:
### Neural Module Networks (NMNs)
- [Neural Module Network](#Neural-Module-Network) **(CVPR 2016)**
- [Dynamic Neural Module Network](#Dynamic-Neural-Module-Network) **(NAACL 2016)**
- [N2NMN (End-to-end Module Network)](#N2NMN) **(ICCV 2017)**
- [Stack-NMN](#Stack-NMN) **(ECCV 2018)**
- [Inferring and Executing Programs for Visual Reasoning](#Inferring-and-Executing-Programs-for-Visual-Reasoning) **(ICCV 2017)**
- [Visual Coreference Resolution in Visual Dialog using Neural Module Networks](#Visual-Coreference-Resolution-in-Visual-Dialog-using-Neural-Module-Networks) **(ECCV 2018)**
- [Transparency by design: Closing the gap between performance and interpretability in visual reasoning](#Transparency-by-design-Closing-the-gap-between-performance-and-interpretability-in-visual-reasoning) **(CVPR 2018)**
- [Explainable and Explicit Visual Reasoning over Scene graphs (CVPR 2019)](#Explainable-and-Explicit-Visual-Reasoning-over-Scene-graphs) **(CVPR 2019)**
- [MAC](#MAC) **(ICLR 2018)**

### Neural Symbolic (\#Todo)
- [Neural-Symbolic VQA](#Neural-Symbolic-VQA) **(NIPS 2018)**
- [The Neuro-Symbolic Concept Learner](#The-Neuro-Symbolic-Concept-Learner) **(ICLR 2019)**

### Graph (\#Todo)
- [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](#Learning-Conditioned-Graph-Structures-for-Interpretable-Visual-Question-Answering) **(NIPS 2018)**
- [Visual Query Answering by Entity-Attribute Graph Matching and Reasoning](#Visual-Query-Answering-by-Entity-Attribute-Graph-Matching-and-Reasoning) **(CVPR 2019)**
- [Neural Task Graphs: Generalizing to Unseen Tasks from a Single Video Demonstration](#Neural-Task-Graphs-Generalizing-to-Unseen-Tasks-from-a-Single-Video-Demonstration)

### Pairwise Relationship (\#Todo)
- [Relation Network](#Relation-Network) **(NIPS 2017)**
- [MUREL Multimodal Relational Reasoning for Visual Question Answering](#MUREL-Multimodal-Relational-Reasoning-for-Visual-Question-Answering) **(CVPR 2019)**

### Survey Paper
- [Compositional models for VQA Can neural module networks really count?](#Compositional-models-for-VQA-Can-neural-module-networks-really-count?)

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
## Stack-NMN

- Accept to ECCV 2018

### Introduction

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Illustration.png" width="600" hegiht="400" align=center/>

*Problems of prior works*
- Existing NMN based approaches need **strong supervision** for decision-making process (predicting module layout)

*Abstract*
- Performs compositional reasoning by automatically inducing a desired sub-task decomposition **without strong supervision**
- Linking different reasoning tasks through **shared modules** that **handle common routines across tasks**
- Experiments show that the proposed method is **more interpretable to human evaluators** compared to other SOTA models

*Comparing to previous NMN works (e.g. N2NMN)*
- Can be trained **without layout supervision**, and replaces the **layout graph with a stack-based data structure** 
- Make the layout **soft and continuous**, can be optimized in a **fully differentiable way using gradient descent**

### Approach

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Framework.png" width="600" hegiht="400" align=center/>

#### Module layout controller
*Layout controller*
- **Decomposes the reasoning task** into a sequence of sub-tasks
- **Translates the input question into a soft layout**, specified via a soft distribution over module weights $w^{(t)}$ at each timestep $t$
- **Supplies each module with a textual parameter $c_t$ at every time-step** using textual attention

*Notations*
- Input question $q$ with $S$ words

*Structure Details*
- Encode the input question $q$ into a $d-$dimensional sequence: 
$$
[h_1, ... , h_S] = BiLSTM(q; \theta_{BiLSTM})
$$
- The controller runs in a recurrent manner from time-step $t=0$ to $t=T-1$
- At each time-step $t$, it applies a time-step dependent linear transform to the question $q$, linearly combines it with the previous $d-$dimensional textual parameter $c_{t-1}$ as:
$$
u = W_2\[W_1^{(t)}q+b_1;c_{t-1}\] + b_2
$$
- Using select the module to execute at current time-step $t$:
$$
w^{(t)} = softmax(MLP(u;\theta_{MLP}))
$$
- Finally, the controller predicts a textual parameter $c_t$ with:
$$
cv_{t,s} = softmax(W_3(u\odot h_s))
$$
$$
c_t = \sum_{s=1}^{S} cv_{t,s}\cdot h_s
$$

#### Neural modules with a memory stack

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Modules.png" width="600" hegiht="400" align=center/>

*Neural modules*
- Each module is a differentiable function with some internal trainable parameters
- Perform specific sub-task
- Can be executed dynamically on-the-fly according to the soft layout
- Mostly following the terminology in N2NMN

*Memory stack*
- To handle **tree-structured layouts** for operations like **compare** or **transform**
- Store and retrieve the intermediate outputs from each module during execution
- Last-In-First-Out (LIFO)
- The push and pop operations are differentiable

#### Training
- Can be trained effectively without reinforcement learning, from task-supervision alone.

*VQA*
- train with softmax cross entropy loss


### Experiments
*Dataset*
- CLEVR, VQA (for VQA)
- CLEVR-Ref (for Referential Expression Grounding)

#### Model performance

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Evaluation-on-CLEVR.png" width="600" hegiht="400" align=center/>

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Evaluation-on-VQAv1VQAv2.png" width="600" hegiht="400" align=center/>

- Joint training the VQA task with REF task can boost the performance for both tasks which shows that the proposed model can **simultaneously handle these two tasks by exploiting the common sub-tasks in them**
- Trained with expert layout still outperforms training from scratch, the gap between the two scenarios is relatively small
- **The model learns to discretely select one module through the soft module weights**
- Comparing to others NMN methods, the proposed Stack-NMN performs comparable to SOTA without supervision of layout prediction


#### Model interpretability
- Human studies on Stack-NMN and MAC

*Metrics*
- subjective perceptions
- truthful beliefs of forward prediction

*Subjective perceptions*

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Visualizations.png" width="600" hegiht="400" align=center/>

- Human evaluators are shown the above image for evaluating
- The Stack-NMN is more clear thant MAC for human evaluators

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Subjective-Perceptions.png" width="600" hegiht="400" align=center/>

*Forward prediction*

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_Stack-NMN_Forward-Prediction.png" width="600" hegiht="400" align=center/>

- Evaluate whether humans can predict the model's answer
- Or detect its failure based on visualizations
- The Stack-NMN is more interpretable than MAC

### Comments
- Time step is a hyper-parameter?
- Why do they combine question information with the previous textual parameter
- RL is better for layout predictions???!
- Why LIFO
- The modules **learn to almost discretely select one module**. This result is interesting
- Evaluated on both of CLEVR and VQA is brave!

---
## Inferring and Executing Programs for Visual Reasoning

- Accept to ICCV 2017

### Introduction
*Problems of previous visual reasoning methods*
- **Black-box** for underlying reasoning process
- Models that learn **direct input-output mappings** tend to **learn dataset biases but not reasoning**

*Proposed method*
- Inspired by Neural Module Network
- Program generator: constructs an **explicit representation** of the **reasoning process**
- Execution engine: executes the resulting program to produce an answer
- Training process: combination of **backpropagation and REINFORCE**

*Comparing with NMN and Dynamic-NMN*
- Does not rely on syntactic parsing or hand-engineered modules
- Only define the function vocabulary and the **universal** module architecture by hand, learning everything else
- Can **generalize to novel questions** by composing modules in ways that not seen during training.

### Approach
#### Overview

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Model.png" width="600" hegiht="400" align=center/>

- Develop a learnable compositional model for VQA
- Input: an image $x$, and a question $q$
- Output: answer $a$ from a fixed set $A$
- Side-results: a program $z$ representing the reasoning steps

*Components*
- Program generator: $z = \pi (q)$, predicts programs from questions
- Execution engine: $a = \phi (x,z)$, executes a program $z$ on an image $x$ to predict an answer $a$

- **The program generator requires very few of supervision programs in order to learn to generalize**

#### Programs
- Focus on learning semantics for a **fixed syntax**
- Pre-specifying a set $F$ of functions $f$
- Represent valid programs $z$ as $syntax$ $trees$ in which each node contains a function $f \in F$

#### Program generator
- Standard LSTM sequence-to-sequence model
- Test time: take the argmax function

#### Execution engine
- Implemented using a neural module network
- Given a program $z$, the execution engine creates a neural network $m(z)$ by mapping each function $f$ to its corresponding module $m_f$ in the order defined by the program
- The output of the child modules are used as input into their corresponding parent module

**Details**
- A module of arity $n$ receives $n$ features maps of shape $C\times H\times W$
- Produces a feature map of shape $C\times H\times W$
- Each unary module is a standard residual block with $3\times 3$ convolutional layers
- The final feature map is flattened and passed into a classifier to predict answers

#### Training
*Semi-supervised learning approach*
- Firstly, use a small set of ground-truth programs to train the program generator
- Then, Fix the generator to train the execution engine using predicted programs
- Finally, use REINFORCE to jointly finetune the program generator and execution engine

### Experiments
*Dataset*
- CLEVR

#### Baselines
- Q-type mode: predicts the most frequent answer
- LSTM: predicts answer uses no image information
- CNN+LSTM: CNN for images and LSTM for questions, features are concatenated and passed to MLP for answer prediction
- CNN+LSTM+SA: Images and Questions features are combined with two rounds of soft spatial attention; a linear transform of the attention output predicts the answer
- CNN+LSTM+SA+MLP: Replace the linear transform with MLP

#### Strongly and semi-supervised learning

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Evaluation-on-CLEVR.png" width="600" hegiht="400" align=center/>

- Trained on strong supervision, the proposed method achieves nearly perfect results
- With about 20K ground-truth programs, the semi-supervised achieves performance near to the strongly supervised one

#### What do the modules learn?

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Visualization.png" width="600" hegiht="400" align=center/>

- The model clearly learns to attend to the correct objects
- The individual modules do in fact **perform their intended functions**
- The modules learn **specialized functions without explicit supervision of their outputs**

#### Generalizing to new attribute combinations

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Compositional-Ability.png" width="600" hegiht="400" align=center/>

- CLEVR-CoGenT dataset for compositional generation
- The results show that actually the proposed model have low compositional abilities; However, only fine-tune on a small set, the model can improve alot

#### Generalizing to new question types

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Unseen-Question-Type.png" width="600" hegiht="400" align=center/>

- Test the ability of proposed model to generalize from short to long chains of reasoning


#### Generalizing to human-posed questions

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_IEP_Human-proposed-dataset.png" width="600" hegiht="400" align=center/>

*CLEVR-Humans Dataset*
- Collect Human-posed questions collected from AMT


### Comments
- "Automatically identifying and learning new modules without program supervision is still an open problem". Charming and horrible problem!
- Justin Johnson is very expert in writing easy-to-read articles!
- "While long-term memory is likely to be crucial component of intelligence, it is not a prerequisite for reasoning, especially the kind of reasoning that is required for answering questions about images"
- How to use the same architecture for all modules
---
## Visual Coreference Resolution in Visual Dialog using Neural Module Networks

- Accept to ECCV 2018

### Introduction

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_CorefNMN_Illustration.png" width="600" hegiht="400" align=center/>

*Abstract*
- Focus on **Visual Coreference Resolution**, involves determining which words, typically **noun phrases and pronouns**, *co-refer* to the same entity/object instance in an image.
- Introducing two novel modules **Refer** and **Exclude** that perform explicit, grounded, coreference resolution at a finer word level.

### Approach
#### Overview

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_CorefNMN_Framework.png" width="600" hegiht="400" align=center/>


*Visual Dialog Definition*
- Input: $H = (C, (Q_1, A_1), ... , (Q_{t-1}, A_{t-1}))$
- Output: ranking a list of answer candidates

*Process*
- **Identifies relevant words or phrases** in the current question that **refer to entities** in the image
- Predicts whether each of these has been mentioned so far
- For novel entities, localize them
- For seen entities, **predicts the relevant coreference** in the history, and **retrieves**
- Use *reference* *pool* to resolve coreferences

*Components*
- Program Generation: reasoning pathway, predicted for the current question $Q_t$
- Program Execution: predicted program is executed by dynamically connecting neural modules to produce a context vector 
- Answer Decoding: the context vector $c_t$ is used to obtain the final answer

#### Neural Modules for Visual Question Answering

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_CorefNMN_Neural-Modules.png" width="600" hegiht="400" align=center/>

*Neural Modules Definition*
- Input image embeddings: x_{vis}, spatial activation maps of image $I$ computed from CNNs
- Input text embeddings: x_{txt}, weighted sum of embeddings of words in the question $Q_t$ using soft attention weights $\alpha$ predicted by a program generator for module $m$
- Input maps: $\{a_i\}$, single-channel spatial maps corresponding to the spatial image embeddings, with number $n_m$
- Module parameters: $\theta_m$
- Neural Module $m$:
$$
y = f_m (x_{vis}, x_{txt}, \lbrace a_i {\rbrace}_{i=1}^{n_m}; \theta_m)
$$
- The output $y$ can be a spatial image attention map $a$ or a context vector $c$
- The output attention map $a$ feeds into next level modules while a context vector $c$ is used to obtain the final answer

*Functions*
- **Find**: localizes objects or attributes, by producing an attention over the image
- **Relocate**: relocate attention for visual relationship related questions, such as "next to", "in front of"
- **And**: Intersection of attention maps
- **Or**: Union of attention maps
- **Describe**: Input an attention map to produce the context vector by describing an attribute 
- **Exist**: Input an attention map to produce the context vector by checking for existence
- **Count**: Input an attention map to produce the context vector by counting

#### Neural Modules for Coreference Resolution
*Reference Pool ($P_{ref}$)*
- To keep track of entities in the dialog
- Keep a semantic dictionary of key-value pairs $\(x_{txt}, a\)$ for all the **Find** modules
- $P_{ref} = \lbrace (x_p^{(i)}, a_p^{(i)}) {\rbrace}_i$


*Refer Module*
- Resolving references in the question $Q_t$, ground them in the conversation history $H$
- Input: text embedding $x_{txt}$, reference pool $P_{ref}$
- Resolves the entity presented by $x_{txt}$ in the form of a soft attention $\alpha$ over $Q_t$
- Step 1 : Learn a *scoring* *network* to compute the similarity between query $x_{txt}$ and candidate $x_p^{(i)}$, introduced a $\Delta_i t$ to measure the distance (considering that the coreference tend to be close to the words in a dialog)
$$
s_i = MLP([x_{txt}, x_p^{(i)}, \Delta_i t])
$$
- Step 2: Apply a softmax function to compute contributions $\tilde{s_i}$ for each entity in the pool
$$
\tilde{s_i} = Softmax(s_i)
$$
- Step 3: weigh the corresponding attention maps to obtain the visual grounding $a_{out}$ for $x_{txt}$
$$
a_{out} = \sum_{i=1}^{\vert P_{ref} \vert}\tilde{s_i} a_p^{(i)}
$$

*Not Module*
- Designed to focus on regions of the image **not** attended by the input attention map $a$

*Exclude Module*
- To handle questions like *"What other red things are present?"*


#### Program Generation
*Purpose*
- Transform the input question to a series of module tokens in order
- Output a soft attention $a_{ti}$ over the question $Q_t$ for every module $m_i$ to instantiate the module
- Decide whether an entity in $Q_t$ has been seen before 

*How*
- An attentional recurrent neural networks with memory

*Details*
- Input: history-aware question encoding $\hat{q_t}$, which is obtained by LSTM over input question, history
- Program decoder: multi-layer LSTM network

#### Other Model Components
*Program execution*
- a simple feed-forward nerual network
- start the computation from the leaf modules 

*Answer Decoding*
- Uses context vector $c_t$ to score answers 

### Experiments
*Datasets*
- Synthetic MNIST Dialog dataset
- VisDial dataset

*MNIST Dialog Dataset*
- Images: $4\*4$ grid of MNIST digits with four attributes: digit class, color, stroke, background color
- Question-Answer pairs: 10 pairs for each image
- Results: proposed CorefNMN achieves near perfect accuracy



*VisDial Dataset*

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_CorefNMN_Evaluation-on-VisDial.png" width="600" hegiht="400" align=center/>

- Images: MS COCO images
- QA pairs: 10-round dialogs for each image
- Results: CorefNMN outperforms all previous methods and at the mean time has relatively high interpretability; Memory augmentation is important; 

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_CorefNMN_Visualization.png" width="600" hegiht="400" align=center/>

### Comments
- Neural Modules actually operates on Attention Maps ???!!!
- Can NMNs be deeper?
---
## Transparency by design-Closing the gap between performance and interpretability in visual reasoning
### Introduction
*Problems with former NMNs methods*
- Are designed with a degree of model transparency, their **performance on complex visual reasoning benchmarks was lacking**

*Goal*
- Close the performance gap between **interpretable models** and *SOTA* visual reasoning methods

*Transparency by Design networks*
- Compose **visual primitives** that leverage an **explicit attention mechanism** to perform **reasoning operations**
- Have the ability to **directly evaluate the model's learning process** via the **produced attention masks**; One can use this to **inspect the semantics of a visual operation**, and **redesign modules** to address apparent aberrations in reasoning

*Definition of Transparency*
- The ability to **examine the intermediate outputs of each module** and **understand their behavior at a high level**

*Contributions*
- Propose a set of **composable visual reasoning primitives** that incorporate an attention mechanism, which allows for model transparency
- SOTA on CLEVR dataset, and on CoGenT generation task
- Show that **compositional visual attention** provides **powerful insight into model behavior**
- Propose a method to **quantitatively evaluate the interpretability of visual attention mechanisms**

*Two main methods in VQA*
- To parse the question into a series of logical operations, then perform each operation over the image features
- To embed both the image and question into a feature space, and then reason over the features jointly

### Approach
#### Overall
*Intuition*
- Combination of IEP, NMN, and N2NMN
- Reuse the primitive operations proposed in [IEP](#Inferring-and-Executing-Programs-for-Visual-Reasoning), but redesign each module according to its intended function in spirit to the [NMN](#Neural-Module-Network) and [N2NMN](#N2NMN)

*Important Design*
- Rather than reine high-dimensional feature maps, a TbD-net passes only **attention masks between its modules, which are one-dimensional attention masks**
- The above idea makes the proposed TbD to be highly interpretable

#### Architecture Details
- Use image features extracted from ResNet-101, and feed these through a simple convolutional block called **stem** which is proposed in [IEP](#Inferring-and-Executing-Programs-for-Visual-Reasoning)
- **Provide stem features to most of our modules**

*Attention*
- Input: image features from the stem and a previous attention to refine
- Output: a heatmap of $1 \times H \times W$
- How: multiplying the input image features and attention mask elementwise, then processing those attended features with a series of convolutions

*And*
- Combine two attention masks in a set intersection

*Or*
- Combine two attention masks in a set union

*Relate*
- Purpose: attends to a region that has some spatial relation to another region
- To relocate the attention
- Problems: need global information from large receptive field
- How: use a series of dilated convolutions to expand the receptive field

*Same*
- Input: stem image features, attention mask
- Output: attention mask
- How: perform a cross-correlation between the object of interest and every other object in the scene to determine which objects share the same property; then send this output through a convolutional layer to produce an attention mask

*Query*
- Purpose: extract a feature from an attended region of an image

*Compare*
- Purpose: compares the properties output by two *Query* modules and produces a feature map 

*Classifier*
- Purpose: takes as input the feature map from either a *Query* or *Compare* module and produces a distribution over answer
- How: followed the work of [IEP](#Inferring-and-Executing-Programs-for-Visual-Reasoning)


### Experiments
*Determines the composition of a modular network*
- Dumplicate to [IEP](#Inferring-and-Executing-Programs-for-Visual-Reasoning)

*Training*
- Use ground truth programs to train the network for program generator

#### CLEVR
*Results*



- Initial results on CLEVR: 98.7%
- Refined: 99.1% on CLEVR
- At the mean time, the proposed method provides straightforward, interpretable outputs at *every stage*
- Using increasing spatial resolution maps results greater performance
- Adding regularization results better attention that there will be less attention on unuseful background

*Transparency*
- Proposed a quantitative analysis of attention, mostly evaluate the Iob
- TbD shows SOTA generative ability.


#### CLEVR-CoGenT
- CLEVR-CoGenT dataset provides test for generalization



### Comments
- Transparency by Design networks seems good!
- Attention maps are not trustful!
- The key idea is the one-dimensional attention maps ????!
- How did they generate the module layout
- Not as much as different from IEP or N2NMN
- Disentangled image representation is important for better generative ablitiy

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
- [Stack NMN](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Ronghang_Hu_Explainable_Neural_Computation_ECCV_2018_paper.pdf)
- [Inferring and Executing Programs for Visual Reasoning](https://arxiv.org/pdf/1705.03633.pdf)
- [Visual Coreference Resolution in Visual Dialog using Neural Module Networks](http://users.ece.cmu.edu/~skottur/papers/corefnmn_eccv18.pdf)
- [Transparency by design: Closing the gap between performance and interpretability in visual reasoning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mascharka_Transparency_by_Design_CVPR_2018_paper.pdf)

- []
- [Learning Conditioned Graph Structures for Interpretable Visual Question Answering](https://arxiv.org/pdf/1806.07243.pdf)




---
