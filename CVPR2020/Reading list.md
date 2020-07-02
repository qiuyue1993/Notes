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

Abstract: We present a novel deep convolutional neural network method for very low bit rate video reconstruction of talking heads. 

Method: The key innovation is a new DCNN architecture that can exploit the audio-video correlations to repair compression defects in the face region. We further improve reconstruction quality by embedding into our DCNN the encoder information of the video compression standards and introducing a constraining projection module in the network.

- Music Gesture for Visual Sound Separation

### Sign Language (Oral:1/1)

- Sign Language Transformers:Joint End-to-End Sign Language Recognition and Translation (**Oral**)

### Text Detection (Recognition) / Scene Text Generation / Scene Text Editing (Oral:3/12)

- UnrealText:Synthesizing Realistic Scene Text Images from the Unreal World

Abstract:  In this paper, we introduce UnrealText, an efficient image synthesis method that renders realistic images via a 3D graphics engine. 3D synthetic engine provides realistic appearance by rendering scene and text as a whole, and allows for better text region proposals with access to precise scene information, e.g. normal and even object meshes. 

- Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection (**Oral**)

Abstract: We propose a novel unified relational reasoning graph network for arbitrary shape text detection.

Method: An innovative local graph bridges a text proposal model via Convolutional Neural Network (CNN) and a deep relational reasoning network via Graph Convolutional Network (GCN), making our network end-to-end trainable. To be concrete, every text instance will be divided into a series of small rectangular components, and the geometry attributes (e.g., height, width, and orientation) of the small components will be estimated by our text proposal model. Given the geometry attributes, the local graph construction model can roughly establish linkages between different text components. For further reasoning and deducing the likelihood of linkages between the component and its neighbors, we adopt a graph-based network to perform deep relational reasoning on local graphs. 

- ABCNet:Real-Time Scene Text Spotting With Adaptive Bezier-Curve Network (**Oral**)

- On Vocabulary Reliance in Scene Text Recognition

- ContourNet:Taking a Further Step Toward Accurate Arbitrary-Shaped Scene Text Detection

- What Machines See Is Not What They Get: Fooling Scene Text Recognition Models With Adversarial Text Images (**Oral**)

- STEFANN:Scene Text Editor Using Font Adaptive Neural Network

- SEED:Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition

- Learn to Augment:Joint Data Augmentation and Network Optimization for Text Recognition

- Fast(er) Reconstruction of Shredded Text Documents via Self-Supervised Deep Asymmetric Metric Learning

- SwapText:Image Based Texts Transfer in Scenes

Abstract: In this work, we present SwapText, a three-stage framework to transfer texts across scene images. SwapText to address a novel task of replacing texts in the scene text images by intended texts.

Method: First, a novel text swapping network is proposed to replace text labels only in the foreground image. Second, a background completion network is learned to reconstruct background images. Finally, the generated foreground image and background image are used to generate the word image by the fusion network. 

Result:  Using the proposing framework, we can manipulate the texts of the input images even with severe geometric distortion. Qualitative and quantitative results are presented on several scene text datasets, including regular and irregular text datasets. 


- OrigamiNet:Weakly-Supervised, Segmentation-Free, One-Step, Full Page Text Recognition by learning to unfold

### Visual Question Answering (VQA) / Text Visual Question Answering (TextVQA) (Oral:4/10)

- Fantastic Answers and Where to Find Them: Immersive Question-Directed Visual Attention

Abstract: We introduce the first dataset of top-down attention in immersive scenes. 

Dataset: The Immersive Questiondirected Visual Attention (IQVA) dataset features visual attention and corresponding task performance (i.e., answer correctness). It consists of 975 questions and answers collected from people viewing 360° videos in a head-mounted display. 

Analyses of the data demonstrate a significant correlation between people’s task performance and their eye movements, suggesting the role of attention in task performance. 

With that, a neural network is developed to encode the differences of correct and incorrect attention and jointly predict the two. The proposed attention model for the first time takes into account answer correctness, whose outputs naturally distinguish important regions from distractions.

- Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing

Problem: Due to deficiencies in models and datasets, today’s models often rely on correlations rather than predictions that are causal w.r.t. data.

Abstract: We propose a novel way to analyze and measure the robustness of the state of the art models w.r.t semantic visual variations as well as propose ways to make models more robust against spurious correlations. Our method performs automated semantic image manipulations and tests for consistency in model predictions to quantify the model robustness as well as generate synthetic data to
counter these problems. 

In addition, we show that models can be made significantly more robust against inconsistent predictions using our edited data.

- Hierarchical Conditional Relation Networks for Video Question Answering (**Oral**)

Abstract: We introduce a general-purpose reusable neural unit called Conditional Relation Network (CRN) that serves as a building block to construct more sophisticated structures for representation and reasoning over video.

Method: A CRN is a relational transformer that encapsulates and maps an array of tensorial objects into a new array of the same kind, conditioned on a contextual feature. Model building becomes a simple exercise of replication, rearrangement and stacking of these reusable units for diverse modalities and contextual information. This design thus supports high-order relational and multi-step reasoning.

The HCRN was evaluated on multiple VideoQA datasets (TGIF-QA, MSVD-QA, MSRVTT-QA)
demonstrating competitive reasoning capability.

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

Abstract: In this paper, we focus on visual navigation in the low-resource setting, where we have only a few training environments annotated with object information.

Method: We propose a novel unsupervised reinforcement learning approach to learn transferable meta-skills (e.g., bypass obstacles, go straight) from unannotated environments without any supervisory signals. The agent can then fast adapt to visual navigation through learning a high-level master policy to combine these metaskills, when the visual-navigation-specified reward is provided.

Experimental results show that our method significantly outperforms the baseline by 53.34% relatively on SPL, and further qualitative analysis demonstrates that our method learns transferable motor primitives for visual navigation.

In our work, we frame low-resource visual navigation as a meta-learning problem. At the metatraining phase, the environments are not annotated with object information, and we assume access to a set of tasks that we refer to as the meta-training tasks. From these tasks, the embodied agent (we call it as meta-learner) then learns a set of transferable sub-policies, each of which corresponds to a specific meta-skill (also called as motor primitives, e.g., bypass obstacles, go straight) by performing a sequence of primitive actions.

- Neural Topological SLAM for Visual Navigation

Abstract: We design topological representations for space that effectively leverage semantics and afford approximate geometric reasoning. At the heart of our representations are nodes with associated semantic features, that are interconnected using coarse geometric information.

Advantages: (a) it uses graphbased representation which allows efficient long-term planning; (b) it explicitly encodes structural priors via function Fs; (c) the geometric function Fg allows efficient exploration and online map building for a new environment; (d) but most importantly, all the functions and policies can be learned in completely supervised manner forgoing the need for unreliable credit assignment via RL.

- Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-Training

- Embodied Language Grounding With 3D Visual Feature Representations

Abstract: We proposed models that associate language utterances with compositional 3D feature representations of the objects and scenes the utterances describe, and exploit the rich constrains of the 3D space for spatial reasoning.

Method:  We infer these 3D visual scene feature maps from RGB images of the scene via view prediction. We present generative models that condition on the dependency tree of an utterance and generate a corresponding visual 3D feature map as well as reason about its plausibility. We also present a detector models that condition on both the dependency tree of an utterance and a related image and localize the object referents in the 3D feature map inferred from the image.

Result: Outperforms 2D-CNN-based methods by a large margin in variety of tasks (classifying plausibility of utterances, detecting referential expressions, supplying rewards for trajectory optimization of object placement policies from language instructions).

- Vision-Language Navigation With Self-Supervised Auxiliary Reasoning Tasks (**Oral**)

Abstract:  We introduce Auxiliary Reasoning Navigation (AuxRN), a framework with four self-supervised auxiliary reasoning tasks to take advantage of the additional training signals derived from the semantic information.
Auxiliary tasks:  explaining the previous actions, estimating the navigation progress, predicting the next orientation, and evaluating the trajectory consistency. Each of the auxiliary tasks exploits useful reasoning knowledge respectively to indicate how an agent understands an environment.

AuxRN pretrained in seen environment with our auxiliary reasoning tasks outperforms our baseline by 3.45% on validation set. Our final model, finetuned on unseen environments with auxiliary reasoning tasks obtains 65%, 4% higher than the previous state-ofthe-art result

- Vision-Dialog Navigation by Exploring Cross-Modal Memory

Abstract: We propose the Cross-modal Memory Network (CMN) for remembering and understanding the rich information relevant to historical navigation actions.

Method:  Our CMN consists of two memory modules, the language memory module (L-mem) and the visual memory module (V-mem). Specifically, L-mem learns latent relationships between the current language interaction and a dialog history by employing a multi-head attention mechanism. V-mem learns to associate the current visual views and the cross-modal memory about the previous navigation actions. The crossmodal memory is generated via a vision-to-language attention and a language-to-vision attention. Benefiting from the collaborative learning of the L-mem and the V-mem, our CMN is able to explore the memory about the decision making of historical navigation actions which is for the current step. 

Results: Experiments on the CVDN dataset show that our CMN outperforms the previous state-of-the-art model by a significant margin on both seen and unseen environments. 

### Image Captioning / Image Captioning Generation (Oral:1/7)

- More Grounded Image Captioning by Distilling Image-Text Matching Model

Abstract: we propose a Part-of-Speech (POS) enhanced image-text matching model - POS-SCAN, as the effective knowledge distillation for more grounded image captioning. We only keep the noun words when computing the matching score with the help of a Part-of-Speech (POS) tagger.

Benifits: given a sentence and an image, POS-SCAN can ground the objects more accurately than SCAN; POS-SCAN serves as a word-region alignment regularization for the captioner’s visual attention module.

Results: conventional image captioners equipped with POS-SCAN can significantly improve the grounding accuracy without strong supervision. 

- Show, Edit and Tell:A Framework for Editing Image Captions

Abstract: This paper proposes a novel approach to image captioning based on iterative adaptive refinement of an existing caption.

Method: Our caption-editing model consisting of two sub-modules: (1) EditNet, a language module with an adaptive copy mechanism (Copy-LSTM) and a Selective Copy Memory Attention mechanism (SCMA), and (2) DCNet, an LSTM-based denoising auto-encoder. 

- Say As You Wish:Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs (**Oral**)

- Normalized and Geometry-Aware Self-Attention Network for Image Captioning

Abstract: We improve Self-attention (SA) from two aspects to promote the performance of image captioning.

Method: First, we propose Normalized Self-Attention (NSA), a reparameterization of SA that brings the benefits of normalization inside SA. Second, to compensate for the major limit of Transformer that it fails to model the geometry structure of the input objects, we propose a class of Geometry-aware Self-Attention (GSA) that extends SA to explicitly and efficiently consider the relative geometry relations between the objects in the image.

Results: We extensively evaluate our proposals on MS-COCO image captioning dataset and superior results are achieved when comparing to state-of-the-art approaches. Further experiments on three challenging tasks, i.e. video captioning, machine translation, and visual question answering, show the generality of our methods.

- Meshed-Memory Transformer for Image Captioning

Abstract: We present  a Meshed Transformer with Memory for Image Captioning. 

Method: The architecture improves both the image encoding and the language generation steps: it learns a multi-level representation of the relationships between image regions integrating learned a priori knowledge, and uses a mesh-like connectivity at decoding stage to exploit low- and high-level features.

Result: When tested on COCO, our proposal achieves a new state of the art in single-model and ensemble configurations on the “Karpathy” test split and on the online test server.  We also assess its performances when describing objects unseen in the training set.

- Better Captioning With Sequence-Level Exploration

- Transform and Tell:Entity-Aware News Image Captioning

Abstract: We propose an end-to-end model which generates captions for images embedded in news articles.

Previous challenges: News image captioning rely on real-worldknowledge, especially about named entities; and they typically have linguistically rich captions that include uncommon words.

Method: We address the first challenge by associating words in the caption with faces and objects in the image, via a multi-modal, multi-head attention mechanism. We tackle the second challenge with a state-of-the-art transformer language model that uses byte-pair-encoding to generate captions as a sequence of word parts.

Result: On the GoodNews dataset,  our model outperforms the previous state of the art by a factor of four in CIDEr score (13 → 54).

We also introduce the NYTimes800k dataset which is 70% larger than GoodNews, has higher article quality, and includes the locations of images within articles as an additional contextual cue.

### Visual Referring Expression (Oral:3/6)

- Graph-Structured Referring Expression Reasoning in the Wild (**Oral**)

- REVERIE:Remote Embodied Visual Referring Expression in Real Indoor Environments (**Oral**)

Abstract: 
Dataset: we propose a dataset of varied and complex robot tasks,　described in　natural language, in terms of objects visible in a large set of real images (Remote Embodied Visual referring Expressions in Real Indoor Environments (REVERIE) task and dataset.). Given an instruction, success requires navigating through a previously-unseen environment to identify an object.

Method: A novel Interactive Navigator-Pointer model is also proposed that provides a strong baseline on the task.

- Multi-Task Collaborative Network for Joint Referring Expression Comprehension and Segmentation (**Oral**)

- Cops-Ref:A New Dataset for Task on Compositional Refering Expression Comprehension

Abstract: We propose a new dataset for visual reasoning in context of referring expression comprehension with two main features.

Features: First, we design a novel expression engine rendering various reasoning logics that can be flexibly combined with rich visual properties to generate expressions with varying compositionality. Second, to better exploit the full reasoning chain embodied in an expression, we propose a new test setting by adding additional distracting images containing objects sharing similar properties with the referent, thus minimising the success rate of reasoning-free.

- Referring Image Segmentation via Cross-Modal Progressive Comprehension

- A Real-Time Cross-Modality Correlation Filtering Method for Referring Expression Comprehension

Abstract: we propose a novel Realtime Cross-modality Correlation Filtering method (RCCF). RCCF reformulates the referring expression comprehension as a correlation filtering process.

Method: The expression is first mapped from the language domain to the visual domain and then treated as a template (kernel) to perform correlation filtering on the image feature map. The peak value in the correlation heatmap indicates the center points of the target box. In addition, RCCF also regresses a 2-D object size and 2-D offset. The center point coordinates, object size and center point offset together to form the target bounding box. 

Result: Our method runs at 40 FPS while achieving leading performance in RefClef, RefCOCO, RefCOCO+ and RefCOCOg benchmarks. In the challenging RefClef dataset, our methods almost double the state-of-the-art performance (34.70% increased to 63.79%).

### Image-Text Matching / Visual-Semantic Matching (Embedding) (5)

- Graph Structured Network for Image-Text Matching

- Multi-Modality Cross Attention Network for Image and Sentence Matching

- Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text

- Visual-Semantic Matching by Exploring High-Order Attention and Distraction

Abstract: We address cross-modality semantic matching task from two previously-ignored aspects: high-order semantic information (e.g., object-predicatesubject triplet, object-attribute pair) and visual distraction (i.e., despite the high relevance to textual query, images may also contain many prominent distracting objects or visual relations). Specifically, we build scene graphs for both visual and textual modalities. 

Technical contributions: We formulate the visual-semantic matching task as an attention-driven cross-modality scene graph matching problem. Graph convolutional networks (GCNs) are used to extract high-order information from two scene graphs; some top-ranked samples are indeed false matching due to the co-occurrence of both highly-relevant and distracting information. We devise an information-theoretic measure for estimating semantic distraction and re-ranking the initial retrieval results.

Result: Comprehensive experiments and ablation studies on two large public datasets (MS-COCO and Flickr30K) demonstrate the superiority of the proposed method and the effectiveness of both high-order attention and distraction.


- MCEN:Bridging Cross-Modal Gap between Cooking Recipes and Dish Images with Latent Variable Model

Abstract: In this paper, we focus on the task of cross-modal retrieval between food images and cooking recipes.

Method:  We present Modality-Consistent Embedding Network (MCEN) that learns modality-invariant representations by projecting images and texts to the same embedding space. To capture the latent alignments between modalities, we incorporate stochastic latent variables to explicitly exploit the interactions between textual and visual features. Importantly, our method learns the cross-modal alignments during training but computes embeddings of different modalities independently at inference time for the sake of efficiency.

Result: Extensive experimental results clearly demonstrate that the proposed MCEN outperforms all existing approaches on the benchmark Recipe1M dataset and requires less computational cost.


### Image-Text Retrieval / Search (3)

- Image Search With Text Feedback by Visiolinguistic Attention Learning

- Context-Aware Attention Network for Image-Text Retrieval

Abstract: In this work, we propose a unified Context-Aware Attention Network (CAAN), which selectively focuses on critical local fragments (regions and words) by aggregating the global context. Specifically, it simultaneously utilizes global intermodal alignments and intra-modal correlations to discover
latent semantic relations. Considering the interactions between images and sentences in the retrieval process, intramodal correlations are derived from the second-order attention of region-word alignments instead of intuitively comparing the distance between original features.

Result: Our method achieves fairly competitive results on two generic imagetext retrieval datasets Flickr30K and MS-COCO.


- IMRAM:Iterative Matching With Recurrent Attention Memory for Cross-Modal Image-Text Retrieval

Fact: Semantics are diverse (i.e. involving different kinds of semantic concepts), and humans usually follow a latent structure to combine them into understandable languages.

Abstract: We propose an Iterative Matching with Recurrent Attention Memory (IMRAM) method, in which correspondences between images and texts are captured with multiple steps of alignments.

Method: We introduce an iterative matching scheme to explore such fine-grained correspondence progressively. A memory distillation unit is used to refine alignment knowledge from early steps to later ones. We formulate the proposed iterative matching method with a recurrent attention memory which incorporates a cross-modal attention unit and a memory distillation unit to refine the correspondence between images and texts.

Result: Experiment results on three benchmark datasets, i.e. Flickr8K, Flickr30K, and MS COCO, show that our IMRAM achieves state-of-the-art performance, well demonstrating its effectiveness. Experiments on a practical business advertisement dataset, named KWAI-AD, further validates the applicability of our method in practical scenarios.

### Image Editing / Manipulation via Text (1)

- ManiGAN: Text-Guided Image Manipulation

### Image generation from text (1)

- RiFeGAN:Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge

### Web Information (3)

- Webly Supervised Knowledge Embedding Model for Visual Reasoning

- Learning From Web Data With Self-Organizing Memory Module

Abstract: In this paper, we propose a novel method, which is capable of handling these two types of noises together (label noise and background noise), without the supervision of clean images in the training stage.

Method: We formulate our method under the framework of multi-instance learning by grouping ROIs (i.e., images and their region proposals) from the same category into bags. ROIs in each bag are assigned with different weights based on the representative/discriminative scores of their nearest clusters, in which the clusters and their scores are obtained via our designed memory module. Our memory module could be naturally integrated with the classification module, leading to an end-to-end trainable system.

- Learning Visual Emotion Representations From Web Data

Abstract: We present a scalable approach for learning powerful visual features for emotion recognition.

Dataset: we curated a webly derived large scale dataset, StockEmotion, which has more than a million images. StockEmotion uses 690 emotion related tags as labels giving us a fine-grained and diverse set of emotion labels, circumventing the difficulty in manually obtaining emotion annotations.

Method: EmotionNet, which we further regularized using joint text and visual embedding and text distillation. We propose methods to handle noisily, partially annotated data, improving visual feature learning through text model distillation and joint visual-text embedding.

Result: EmotionNet trained on the StockEmotion dataset outperforms SOTA models on four different visual emotion tasks. EmotionNet achieves competitive zero-shot recognition performance against fully supervised baselines on a challenging visual emotion dataset, EMOTIC.

### Miscellaneous (Oral:1/10)

- Counterfactual Vision and Language Learning (**Oral**)

Existing problem: It is particularly remarkable that this success has been achieved on the basis of comparatively small datasets, given the scale of the problem. One explanation is that this has been accomplished partly by exploiting bias in the datasets rather than developing deeper multi-modal reasoning. This fundamentally limits the generalization of the method, and thus its practical applicability.

Propose: We propose a method that addresses this problem by introducing counterfactuals in the training. In doing so we leverage structural causal models for counterfactual evaluation to formulate alternatives, for instance, questions that could be asked of the same image set. We encourage the model to reason about “what the answer could be about a counterfactual image or question”.


Result: We show that simulating plausible alternative training data through this process results in better generalization.

- Learning Unseen Concepts via Hierarchical Decomposition and Composition

Abstract: We present a hierarchical decompositionand-composition (HiDC) model for unseen compositional concept recognition.

Method: We propose to decompose each seen image as visual elements and learn the corresponding subconcepts in independent subspaces. We generate compositions from these subspaces in three hierarchical forms, and learn the composed concepts in a unified composition space. We define semi-positive concepts to depict finegrained contextual relationships between sub-concepts, and learn accurate compositional concepts with adaptive pseudo supervision exploited from the generated compositions. 

Result: We validate the proposed approach on two challenging benchmarks, and demonstrate its superiority over state-of-the-art approaches.


- 12-in-1:Multi-Task Vision and Language Representation Learning

- ALFRED:A Benchmark for Interpreting Grounded Instructions for Everyday Tasks

Abstract: We present ALFRED (Action Learning From Realistic Environments and Directives), a benchmark for learning a mapping from natural language instructions and egocentric vision to sequences of actions for household tasks.

ALFRED consists of expert demonstrations in interactive visual environments for 25k natural language directives. These directives contain both high-level goals like “Rinse off a mug and place it in the coffee maker.” and low-level language instructions like “Walk to the coffee maker on the right.”

The long horizon of ALFRED tasks poses a significant challenge with sub-problems including visual semantic navigation, object detection, referring expression grounding, and action grounding. These challenges may be approachable by models that exploit hierarchy, modularity,
and structured reasoning and planning.

- Visual Commonsense R-CNN

Abstract: We present a novel unsupervised feature representation learning method, Visual Commonsense Region-based Convolutional Neural Network (VC R-CNN), to serve as an improved visual region encoder for high-level tasks such as captioning and VQA.

Method: The prediction of VC R-CNN is by using causal intervention: P(Y |do(X)), while others are by using the conventional likelihood: P(Y |X). 

- Active Speakers in Context

Existing problem: Current methods for active speaker detection focus on modeling short-term audiovisual information from a single speaker. Although this strategy can be enough for addressing single-speaker scenarios, it prevents accurate detection when the task is to identify who of many candidate speakers are talking.

Propose: This paper introduces the Active Speaker Context, a novel representation that models relationships between multiple speakers over long time horizons. Our
Active Speaker Context is designed to learn pairwise and temporal relations from an structured ensemble of audiovisual observations.

- MMTM:Multimodal Transfer Module for CNN Fusion

- Hierarchical Graph Attention Network for Visual Relationship Detection

Previous problems: Existing graph-based methods mainly represent the relationships by an object-level graph, which ignores to model the triplet level dependencies.

Abstract: In this work, a Hierarchical Graph Attention Network (HGAT) is proposed to capture the dependencies on both object-level and triplet-level.

Method: Object level graph aims to capture the interactions between objects, while the triplet-level graph models the dependencies among relation triplets. In addition, prior knowledge and attention mechanism are introduced to fix the redundant or missing edges on graphs that are constructed according to spatial correlation.

Result: Experimental results on the well-known VG and VRD datasets demonstrate that our model significantly outperforms the state-of-the-art methods.


- Discriminative Multi-Modality Speech Recognition

Abstract: We propose a two-stage speech recognition model. In the first stage, the target voice is separated from background noises with help from the corresponding visual information of lip movements, making the model ‘listen’ clearly. At the second stage, the audio modality combines visual modality again to better understand the speech by a MSR sub-network, further improving the recognition rate. 

Key contributions: we introduce a pseudo-3D residual convolution (P3D)-based visual front-end to extract more discriminative features; we upgrade the temporal convolution block from 1D ResNet with the temporal convolutional network (TCN), which is more suitable for the temporal tasks;  the MSR sub-network
is built on the top of Element-wise-Attention Gated Recurrent Unit (EleAtt-GRU), which is more effective than Transformer in long sequences.

Result: We conducted extensive experiments on the LRS3-TED and the LRW datasets. Our twostage model (audio enhanced multi-modality speech recognition, AE-MSR) consistently achieves the state-of-the-art performance by a significant margin, which demonstrates the necessity and effectiveness of AE-MSR.

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

Abstract: Holistically understanding an object with its 3D movable parts is essential for visual models of a robot to interact with the world, we fill this important missing piece in autonomous driving by solving two critical issues.

Contents: We propose an effective training data generation process by fitting a 3D car model with dynamic parts to cars in real images; we collected a large dataset in real driving scenario with cars in uncommon states (CUS), i.e. with door or trunk opened etc.

Method: We propose a multi-task deep network which can simultaneously output 2D detection, instance-level segmentation, dynamic part segmentation, and state description.

Result: The experimental results show that our editing data and deep
network perform well on CUS.

---
## Scene Analysis and Understanding (2)

- Weakly Supervised Visual Semantic Parsing

Previous problems: Existing SGG methods require millions of manually annotated bounding boxes for training, and are computationally inefficient, as they exhaustively process all pairs of object proposals to detect predicates.

Abstract: Proposing a generalized formulation of SGG, namely Visual Semantic Parsing, which disentangles entity and predicate recognition, and enables sub-quadratic performance. Then we propose the Visual Semantic Parsing Network, VSPNET, based on a dynamic, attention-based, bipartite message passing framework that jointly infers graph nodes and edges through an iterative process. Additionally, we propose the first graphbased weakly supervised learning framework, based on a novel graph alignment algorithm, which enables training without bounding box annotations.

- Learning 3D Semantic Scene Graphs From 3D Indoor Reconstructions

Abstract:  We leverage inference on scene graphs as a way to carry out 3D scene understanding, mapping objects and their relationships.

Method: In particular, we propose a learned method that regresses a scene graph from the point cloud of a scene. Our novel architecture is based on PointNet and Graph Convolutional Networks (GCN). In addition, we introduce 3DSSG, a semiautomatically generated dataset, that contains semantically rich scene graphs of 3D scenes.

Result: We show the application of our method in a domain-agnostic retrieval task, where graphs serve as an intermediate representation for 3D-3D and 2D-3D matching.



---
## Others (1)

- Online Knowledge Distillation via Collaborative Learning

Abstract: KDCL, which is able to consistently improve the generalization ability of deep neural networks (DNNs) that have different learning capacities.

KDCL treats all DNNs as “students” and collaboratively trains them in a single stage (knowledge is transferred among arbitrary students during
collaborative training), enabling parallel computing, fast computations, and appealing generalization ability. Specifically, we carefully design multiple methods to generate soft target as supervisions by effectively ensembling predictions of students and distorting the input images.

---
## Notes

- Web data

- Robotics, V and L

- VLN SLAM

- Edit

- No 3D? Or 3D = Embodied

- Video + Language = a lot

- Text: a lot

- Graph NN -> Relationships

- Grounding vs. Captioning

- Knowledge Distillation

- Transformer

- Causal Reasoning

