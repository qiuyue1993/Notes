# Reading List

## Indexing
- [Awarded Papers](#Awarded-Papers)
- [Language and Vision](#Language-and-Vision)
- [Image Synthesis](#Image-Synthesis)
- [3D Vision](#3D-Vision)
- [Scene Analysis and Understanding](#Scene-Analysis-and-Understanding)
- [Others](#Others)
---
## Awarded Papers

- BSP-Net: Generating Compact Meshes via Binary Space Partitioning (**Best Student Paper**)

Abstract: We are inspired by a classical spatial data structure from computer graphics, Binary Space Partitioning (BSP), to facilitate 3D learning. The core ingredient of BSP is an operation for recursive subdivision of space to obtain convex sets. 

BSP-Net: BSP-Net, an unsupervised method which can generate compact and structured polygonal meshes in the form of convex decomposition. The convexes inferred by BSPNet can be easily extracted to form a polygon mesh, without any need for iso-surfacing.

- DeepCap:Monocular Human Performance Capture Using Weak Supervision (**Best Student Paper Honorable Mention**)

Abstract: We propose a novel deep learning approach for monocular dense human performance capture. Our method is trained
in a weakly supervised manner based on multi-view supervision completely removing the need for training data with 3D ground truth annotations. The network architecture is based on two separate networks that disentangle the task into a pose estimation and a non-rigid surface deformation step. 

• A learning-based 3D human performance capture approach that jointly tracks the skeletal pose and the nonrigid surface deformations from monocular images.
• A new differentiable representation of deforming human surfaces which enables training from multi-view
video footage directly.

- UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders (**Paper award nominees**)

Abstract:　Inspired by the saliency data labeling process, we propose　probabilistic RGB-D saliency detection network via conditional variational autoencoders to model human annotation uncertainty and generate multiple saliency maps for each input image by sampling in the latent space. 

Quantitative and qualitative evaluations on six challenging benchmark datasets against 18 competing algorithms demonstrate the effectiveness of our approach in
learning the distribution of saliency maps, leading to a new state-of-the-art in RGB-D saliency detection.

- Transferring Cross-domain Knowledge for Video Sign Language Recognition (**Paper award nominees**)

Abstract: We observe that despite the existence of large domain gaps, isolated and news signs share the same visual concepts, such as hand gestures and body movements. Motivated by this observation, we propose a novel method that
learns domain-invariant descriptors and fertilizes WSLR models by transferring knowledge of subtitled news sign to them.

Method: We extract news signs using a base WSLR model, and then design a classifier jointly trained on news and isolated signs to coarsely align these two domains. In order to learn domain-invariant features within each class and suppress domain-specific features, our method further resorts to an external memory to store the class centroids of the aligned news signs. We then design a temporal attention based on the learnt descriptor to improve recognition performance. 

- Cross-Batch Memory for Embedding Learning (**Paper award nominees**)

Abstract: 

Slow drift phenomena: The embedding features drift exceptionally slow even as the model parameters are updating throughout the training process.

XBM: We propose a cross-batch memory (XBM) mechanism that memorizes the embeddings of past iterations, allowing the model to collect sufficient hard negative pairs across multiple mini-batches - even over the whole dataset.

Our XBM can be directly integrated into a general pairbased DML framework, where the XBM augmented DML can boost performance considerably.

We have presented a conceptually simple, easy to implement, and memory efficient cross-batch mining mechanism for pair-based DML.

- TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning (**Paper award nominees**)

Abstract: In this work, we propose a progressive texture-fusion method specially designed for real-time RGB-D scanning.

Method: To this end, we first devise a novel texture-tile voxel grid, where texture tiles are embedded in the voxel grid of the signed distance function, allowing for high-resolution texture mapping on the low-resolution geometry volume. We associate vertices of implicit geometry directly with texture coordinates.  Second, we introduce real-time texture warping that applies a spatiallyvarying perspective mapping to input images so that texture warping efficiently mitigates the mismatch between the intermediate geometry and the current input view.

- Total3DUnderstanding:Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image (**Paper award nominees**)

Abstract: We propose an end-to-end solution to jointly reconstruct room layout, object bounding boxes and meshes from a single image. 

Method: a coarse-to-fine hierarchy with three components: 1. room layout with camera pose; 2. 3D object bounding boxes; 3. object meshes.

---
## Language and Vision (Oral:18/89)

### Video / Action and Language (Oral:1/16)

- Visual-Textual Capsule Routing for Text-Based Video Segmentation (**Oral**)

- Modality Shifting Attention Network for Multi-Modal Video Question Answering

- Dense Regression Network for Video Grounding

- Video Object Grounding Using Semantic Roles in Language Description

- Fine-Grained Video-Text Retrieval With Hierarchical Graph Reasoning

- Where Does It Exist:Spatio-Temporal Video Grounding for Multi-Form Sentences

- Local-Global Video-Text Interactions for Temporal Grounding

- Visual Grounding in Video for Unsupervised Word Translation

- Spatio-Temporal Graph for Video Captioning With Knowledge Distillation

- Straight to the Point:Fast-Forwarding Videos via Reinforcement Learning Using Textual Data

- Violin:A Large-Scale Dataset for Video-and-Language Inference

- Syntax-Aware Action Targeting for Video Captioning

- Object Relational Graph With Teacher-Recommended Learning for Video Captioning

- Speech2Action: Cross-Modal Supervision for Action Recognition

- Listen to Look:Action Recognition by Previewing Audio

- Beyond Short-Term Snippet:Video Relation Detection With Spatio-Temporal Global Context

### Audio and Vision (Oral:1/2)

- DAVD-Net:Deep Audio-Aided Video Decompression of Talking Heads (**Oral**)

- Music Gesture for Visual Sound Separation

### Sign Language (Oral:1/1)

- Sign Language Transformers:Joint End-to-End Sign Language Recognition and Translation (**Oral**)

### Text Detection (Recognition) / Scene Text Generation / Scene Text Editing (Oral:3/12)

- UnrealText:Synthesizing Realistic Scene Text Images from the Unreal World

Abstract:  In this paper, we introduce UnrealText, an efficient image synthesis method that renders realistic images via a 3D graphics engine. 3D synthetic engine provides realistic appearance by rendering scene and text as a whole, and allows for better text region proposals with access to precise scene information, e.g. normal and even object meshes. 


- Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection (**Oral**)

- ABCNet:Real-Time Scene Text Spotting With Adaptive Bezier-Curve Network (**Oral**)

- On Vocabulary Reliance in Scene Text Recognition

- ContourNet:Taking a Further Step Toward Accurate Arbitrary-Shaped Scene Text Detection

- What Machines See Is Not What They Get: Fooling Scene Text Recognition Models With Adversarial Text Images (**Oral**)

- STEFANN:Scene Text Editor Using Font Adaptive Neural Network

- SEED:Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition

- Learn to Augment:Joint Data Augmentation and Network Optimization for Text Recognition

- Fast(er) Reconstruction of Shredded Text Documents via Self-Supervised Deep Asymmetric Metric Learning

- SwapText:Image Based Texts Transfer in Scenes

- OrigamiNet:Weakly-Supervised, Segmentation-Free, One-Step, Full Page Text Recognition by learning to unfold

### Visual Question Answering (VQA) / Text Visual Question Answering (TextVQA) (Oral:4/10)

- Fantastic Answers and Where to Find Them: Immersive Question-Directed Visual Attention

- Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing

- Hierarchical Conditional Relation Networks for Video Question Answering (**Oral**)

- Iterative Answer Prediction With Pointer-Augmented Multimodal Transformers for TextVQA (**Oral**)

- SQulNTing at VQA Models:Introspecting VQA Models With Sub-Questions (**Oral**) (**Liked**)
Problems: existing VQA models have consistency issues – they answer the reasoning question correctly but fail on associated lowlevel perception questions.

Abstract: We quantify the extent to which this phenomenon occurs by creating a new Reasoning split of the VQA dataset and collecting Sub-VQA, a new dataset1 consisting of 200K new perception questions which serve as sub questions corresponding to the set of perceptual tasks needed to effectively answer the complex reasoning questions in the Reasoning split.

Method: we propose an approach called Sub-Question Importance-aware Network Tuning (SQuINT), which encourages the model to attend do the same parts of the image when answering the reasoning question and the perception sub questions. We show that SQuINT improves model consistency by 7.8%, also marginally improving its performance on the Reasoning questions in VQA, while also displaying qualitatively better attention maps.



- TA-Student VQA:Multi-Agents Training by Self-Questioning (**Oral**)

- On the General Value of Evidence, and Bilingual Scene-Text Visual Question Answering

- In Defense of Grid Features for Visual Question Answering

- VQA With No Questions-Answers Training

- Counterfactual Samples Synthesizing for Robust Visual Question Answering

### Visual Reasoning (2)

- Differentiable Adaptive Computation Time for Visual Reasoning

- Gold Seeker:Information Gain From Policy Distributions for Goal-Oriented Vision-and-Language Reasoning

### Visual Dialog (Oral:1/2)

- Iterative Context-Aware Graph Inference for Visual Dialog (**Oral**)

Abstract: We propose a novel Context-Aware Graph (CAG) neural network. Each node in the graph corresponds to a joint semantic feature, including both objectbased (visual) and history-related (textual) context representations. 

The graph structure (relations in dialog) is iteratively updated using an adaptive top-K message passing mechanism. Then, after the update, we impose graph attention on all the nodes to get the final graph embedding and infer the answer.

Experimental results on VisDial v0.9 and v1.0 datasets show that CAG outperforms comparative methods. Visualization results further validate the interpretability of our method.

- Two Causal Principles for Improving Visual Dialog

Abstract: This paper unravels two causal principlesfor improving Visual Dialog (VisDial).

Principle 1: we should remove the direct input of the dialog history to the answer model, otherwise a harmful shortcut bias will be introduced.

Principle 2: There is an unobserved confounder for history, question, and answer, leading to spurious correlations from training data.

The two principles are model-agnostic, so they are applicable in any VisDial model. 



### Embodied AI / Vision-Language Navigation (VLN) (Oral:2/8)

- RoboTHOR:An Open Simulation-to-Real Embodied AI Platform

- SAPIEN:A SimulAted Part-Based Interactive ENvironment (**Oral**)
Abstract: We present SAPIEN, a simulation environment for robotic vision and interaction tasks, which provides detailed part-level physical simulation, hierarchical robotics controllers and versatile rendering options.
SAPIEN: PhysX-based interaction-rich and physics-realistic simulation environment; SAPIEN Asset: Including PartNet Mobility dataset, conating 14K movable parts over 2,346 3D models (46 common indoor object categories); SAPIEN Renderer: fast-frame-rate OpenGL rasterizer and more photorelistic ray-tracing options.

Benckmark: We evaluate stateof-the-art vision algorithms for part detection and motion attribute recognition as well as demonstrate robotic interaction tasks using heuristic approaches and reinforcement learning algorithms.


- Unsupervised Reinforcement Learning of Tranferable Meta-Skills for Embodied Navigation

- Neural Topological SLAM for Visual Navigation

- Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-Training

- Embodied Language Grounding With 3D Visual Feature Representations 

- Vision-Language Navigation With Self-Supervised Auxiliary Reasoning Tasks (**Oral**)

Abstract:  We introduce Auxiliary Reasoning Navigation (AuxRN), a framework with four self-supervised auxiliary reasoning tasks to take advantage of the additional training signals derived from the semantic information.
Auxiliary tasks:  explaining the previous actions, estimating the navigation progress, predicting the next orientation, and evaluating the trajectory consistency. Each of the auxiliary tasks exploits useful reasoning knowledge respectively to indicate how an agent understands an environment.

AuxRN pretrained in seen environment with our auxiliary reasoning tasks outperforms our baseline by 3.45% on validation set. Our final model, finetuned on unseen environments with auxiliary reasoning tasks obtains 65%, 4% higher than the previous state-ofthe-art result

- Vision-Dialog Navigation by Exploring Cross-Modal Memory

### Image Captioning / Image Captioning Generation (Oral:1/7)

- More Grounded Image Captioning by Distilling Image-Text Matching Model

- Show, Edit and Tell:A Framework for Editing Image Captions

Abstract: This paper proposes a novel approach to image captioning based on iterative adaptive refinement of an existing caption.

Method: Our caption-editing model consisting of two sub-modules: (1) EditNet, a language module with an adaptive copy mechanism (Copy-LSTM) and a Selective Copy Memory Attention mechanism (SCMA), and (2) DCNet, an LSTM-based denoising auto-encoder. 


- Say As You Wish:Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs (**Oral**)

- Normalized and Geometry-Aware Self-Attention Network for Image Captioning

- Meshed-Memory Transformer for Image Captioning

- Better Captioning With Sequence-Level Exploration

- Transform and Tell:Entity-Aware News Image Captioning

### Visual Referring Expression (Oral:3/6)

- Graph-Structured Referring Expression Reasoning in the Wild (**Oral**)

- REVERIE:Remote Embodied Visual Referring Expression in Real Indoor Environments (**Oral**)

Abstract: 
Dataset: we propose a dataset of varied and complex robot tasks, described in
natural language, in terms of objects visible in a large set of real images (Remote Embodied Visual referring Expressions in Real Indoor Environments (REVERIE) task and dataset.). Given an instruction, success requires navigating through a previously-unseen environment to identify an object.
Method: A novel Interactive Navigator-Pointer model is also proposed that provides a strong baseline on the task.


- Multi-Task Collaborative Network for Joint Referring Expression Comprehension and Segmentation (**Oral**)

- Cops-Ref:A New Dataset for Task on Compositional Refering Expression Comprehension

- Referring Image Segmentation via Cross-Modal Progressive Comprehension

- A Real-Time Cross-Modality Correlation Filtering Method for Referring Expression Comprehension

### Image-Text Matching / Visual-Semantic Matching (Embedding) (5)

- Graph Structured Network for Image-Text Matching

- Multi-Modality Cross Attention Network for Image and Sentence Matching

- Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text

- Visual-Semantic Matching by Exploring High-Order Attention and Distraction

- MCEN:Bridging Cross-Modal Gap between Cooking Recipes and Dish Images with Latent Variable Model

### Image-Text Retrieval / Search (3)

- Image Search With Text Feedback by Visiolinguistic Attention Learning

- Context-Aware Attention Network for Image-Text Retrieval

- IMRAM:Iterative Matching With Recurrent Attention Memory for Cross-Modal Image-Text Retrieval

### Image Editing / Manipulation via Text (1)

- ManiGAN: Text-Guided Image Manipulation

### Image generation from text (1)

- RiFeGAN:Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge

### Web Information (3)

- Webly Supervised Knowledge Embedding Model for Visual Reasoning

- Learning From Web Data With Self-Organizing Memory Module

- Learning Visual Emotion Representations From Web Data

### Miscellaneous (Oral:1/10)

- Counterfactual Vision and Language Learning (**Oral**)

Existing problem: It is particularly remarkable that this success has been achieved on the basis of comparatively small datasets, given the scale of the problem. One explanation is that this has been accomplished partly by exploiting bias in the datasets rather than developing deeper multi-modal reasoning. This fundamentally limits the generalization of the method, and thus its practical applicability.

Propose: We propose a method that addresses this problem by introducing counterfactuals in the training. In doing so we leverage structural causal models for counterfactual evaluation to formulate alternatives, for instance, questions that could be asked of the same image set. We encourage the model to reason about “what the answer could be about a counterfactual image or question”.


Result: We show that simulating plausible alternative training data through this process results in better generalization.



- Learning Unseen Concepts via Hierarchical Decomposition and Composition

- 12-in-1:Multi-Task Vision and Language Representation Learning

- ALFRED:A Benchmark for Interpreting Grounded Instructions for Everyday Tasks

- Visual Commonsense R-CNN

Abstract: We present a novel unsupervised feature representation learning method, Visual Commonsense Region-based Convolutional Neural Network (VC R-CNN), to serve as an improved visual region encoder for high-level tasks such as captioning and VQA.

Method: The prediction of VC R-CNN is by using causal intervention: P(Y |do(X)), while others are by using the conventional likelihood: P(Y |X). 

- Active Speakers in Context

Existing problem: Current methods for active speaker detection focus on modeling short-term audiovisual information from a single speaker. Although this strategy can be enough for addressing single-speaker scenarios, it prevents accurate detection when the task is to identify who of many candidate speakers are talking.

Propose: This paper introduces the Active Speaker Context, a novel representation that models relationships between multiple speakers over long time horizons. Our
Active Speaker Context is designed to learn pairwise and temporal relations from an structured ensemble of audiovisual observations.

- MMTM:Multimodal Transfer Module for CNN Fusion

- Hierarchical Graph Attention Network for Visual Relationship Detection

- Discriminative Multi-Modality Speech Recognition

- PhraseCut: Language-Based Image Segmentation in the Wild

---
## Image Synthesis (3)

- Semantic Image Manipulation Using Scene Graphs

- Semantically Multi-Modal Image Synthesis

- SynSin:End-to-End View Synthesis From a Single Image

---
## 3D Vision (2)

- Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes From a Single Image (**Oral**)

- 3D Part Guided Image Editing for Fine-Grained Object Understanding

---
## Scene Analysis and Understanding (2)

- Weakly Supervised Visual Semantic Parsing

- Learning 3D Semantic Scene Graphs From 3D Indoor Reconstructions

---
## Others (1)

- Online Knowledge Distillation via Collaborative Learning


---
## Notes

- Web data

- Video

- Robotics, V and L

- VLN SLAM

- Edit

- No 3D? Or 3D = Embodied

- Video + Language = a lot

- Text: a lot


