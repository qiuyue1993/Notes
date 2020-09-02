# Reading list

## TextCaps: a Dataset for Image Captioning with Reading Comprehension

Existing problem: Current approaches are unable to include written text in their descriptions, although text is omnipresent in human environments and frequently critical to understand our surroundings. 

Propose dataset: To study how to comprehend text in the context of an image we collect a novel dataset, TextCaps, with 145k captions for 28k images. Our dataset challenges a model to recognize text, relate it to its visual context, and decide what part of the text to copy or paraphrase, requiring spatial, semantic, and visual reasoning between multiple text tokens and visual entities, such as objects.

Our analysis also points out several challenges of this dataset: Different from other captioning datasets, nearly all our captions require integration of OCR tokens, many are unseen (“zero-shot”). In contrast to TextVQA datasets, TextCaps requires generating long sentences and involves new technical challenges, including many switches between OCR and vocabulary tokens.

Experiment:  We study baselines and adapt existing approaches to this new task, which we refer to as image captioning with reading comprehension. Our analysis with automatic and human studies shows that our new TextCaps dataset provides many new technical challenges over previous datasets.


## Object-and-Action Aware Model for Visual Language Navigation

Existing problems: most existing methods pay few attention to distinguish these information from each other during instruction encoding and mix together the matching between textual object/action encoding and visual perception/orientation features of candidate viewpoints.

Proposed method: We propose an Object-and-Action Aware Model (OAAM) that processes these two different forms of natural language based instruction separately. 

However, one side-issue caused by above solution is that an object mentioned in instructions may be observed in the direction of two or more candidate viewpoints, thus the OAAM may not predict the viewpoint on the shortest path as the next action. To handle this problem, we design a simple but effective path loss to penalize trajectories deviating from the ground truth path. To encourage the robot agent to stay on the path, we additional propose a path loss based on the distance to nearest ground-truth viewpoint.

Result: Experimental results demonstrate the effectiveness of the proposed model and path loss, and the superiority of their combination with a 50% SPL score on the R2R dataset and a 40% CLS score on the R4R dataset in unseen environments, outperforming the previous state-of-the-art.

## ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

- We introduce the task of 3D object localization in RGB-D scans using natural language descriptions. As input, we assume a point cloud of a scanned 3D scene along with a free-form description of a specified target object. 

-  To address this task, we propose ScanRefer, learning a fused descriptor from 3D object proposals and encoded sentence embeddings. This fused descriptor correlates language expressions with geometric features, enabling regression of the 3D bounding box of a target object. 

- We also introduce the ScanRefer dataset, containing 51, 583 descriptions of 11, 046 objects from 800 ScanNet scenes. ScanRefer is the first large-scale effort to perform object localization via natural language expression directly in 3D. 

- We show that our end-to-end method outperforms the best 2D visual grounding method that simply backprojects its 2D predictions to 3D by a significant margin (9.04 Acc@0.5IoU vs. 22.39 Acc@0.5IoU).

## Active Visual Information Gathering for Vision-Language Navigation

Existing problems: Agents trained by current approaches typically suffer from this and would consequently struggle to avoid random and inefficient actions at every step.

Inspiration: In contrast, when humans face such a challenge, they can still maintain robust navigation by actively exploring the surroundings to gather more information and thus make more confident navigation decisions.  This work draws inspiration from human navigation behavior and endows an agent with an active information gathering ability for a more intelligent vision-language navigation policy.

Proposed method: y. To achieve this, we propose an end-to-end framework for learning an exploration policy that decides i) when and where to explore, ii) what information is worth gathering during exploration, and iii) how to adjust the navigation decision after the exploration.

Results:  The experimental results show promising exploration strategies emerged from training, which leads to significant boost in navigation performance. On the R2R challenge leaderboard, our agent gets promising results all three VLN settings, i.e., single run, pre-exploration, and beam search.

## Environment-agnostic Multitask Learning for Natural Language Grounded Navigation

Existing problem: Existing methods tend tooverfit training data in seen environments and fail to generalize well in previously unseen environments. 

Proposed method: To close the gap between seen and unseen environments, we aim at learning a generalized navigation model from two novel perspectives: (1) we introduce a multitask navigation model that can be seamlessly trained on both Vision-Language Navigation (VLN) and Navigation from Dialog History (NDH) tasks, which benefits from richer natural language guidance and effectively transfers knowledge across tasks; (2) we propose to learn environment-agnostic representations for the navigation policy that are invariant among the environments seen during training, thus generalizing better on unseen environments. 

In this work, we presented an environment-agnostic multitask learning framework to learn generalized policies for agents tasked with natural language grounded navigation.

Results: Extensive experiments show that environment-agnostic multitask learning significantly reduces the performance gap between seen and unseen environments, and the navigation agent trained so outperforms baselines on unseen environments by 16% (relative measure on success rate) on VLN and 120% (goal progress) on NDH. Our submission to the CVDN leaderboard establishes a new state-of-the-art for the NDH task on the holdout test set.

## Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments

Proposed task: We develop a language-guided navigation task set in a continuous 3D environment where agents must execute low-level actions to　follow natural language navigation directions.

Difference with VLN: Specifically, our setting drops the presumptions of known environment topologies, short-range oracle navigation, and perfect agent localization. This suggests prior results in VLN may be overly optimistic in terms of progress towards instruction-following robots functioning in the wild.

Experiment: To contextualize this new task, we develop models that mirror many of the advances made in prior settings as well as single-modality baselines. While some transfer, we find significantly lower absolute performance in the continuous setting – suggesting that performance in prior ‘navigation-graph’ settings may be inflated by the strong implicit assumptions.

## Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler

Problem in VLN: One of the problems of the VLN task is data scarcity since it is difficult to collect enough navigation paths with human-annotated instructions for interactive environments. 

Proposed method: We integrate counterfactual thinking into the vision-and-language navigation task, and propose the adversarial path sampler (APS) to progressively sample challenging and effective paths to improve the navigation policy. We propose an adversarial-driven counterfactual reasoning model that can consider effective conditions instead of low-quality augmented data. In particular, we present a model-agnostic adversarial path sampler (APS) that learns to sample challenging paths that force the navigator to improve based on the navigation performance. APS also serves to do pre-exploration of unseen environments to strengthen the model’s ability to generalize. 

Experiment and result: We evaluate the influence of APS on the performance of different VLN baseline models using the room-to-room dataset (R2R). The results show that the adversarial training process with our proposed APS benefits VLN models under both seen and unseen environments. And the pre-exploration process can further gain additional improvements under unseen environments.

## Soft Expert Reward Learning for Vision-and-Language Navigation

Previous problems: Dominant methods based on supervised learning clone expert’s behaviours and thus perform better on seen environments, while showing restricted performance on unseen ones. Reinforcement Learning (RL) based models show better generalisation ability but have issues as well, requiring large amount of manual reward engineering is
one of which. In this paper, we propose a Soft Expert Reward Learning (SERL) model to address the behaviour cloning error accumulation and the reinforcement learning reward engineering issues for VLN task.

Proposed method: Wwe introduce a Soft Expert Reward Learning (SERL) model to overcome the reward engineering designing and generalisation problems of the VLN task. Our proposed method consists of two complementary components: Soft Expert Distillation (SED) module encourages agents to behave like an expert as much as possible, but in a soft fashion (In this paper we introduce a Soft Expert Reward Learning model to distil reward function directly from expert demonstrations and soften the process of behaviour cloning to alleviate the drawbacks from error accumulation. ); Self Perceiving (SP) module targets at pushing the agent towards the final destination as fast as possible.

Result: From the experimental results, we show that our SERL model gains better performance generally than current state-of-the-art methods in both validation unseen and test unseen set on VLN Room-to-Room dataset. The ablation study shows that our proposed the Soft Expert Distillation (SED) module and the Self Perceiving (SP) module are complementary to each other. Moreover, the visualisation experiments further verify the SERL model can overcome the error accumulation problem. 

## Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models

Previous problem: Recent Transformer-based large-scale pre-trained models have revolutionized vision-and-language (V+L) research. Models such as ViLBERT, LXMERT and UNITER have significantly lifted state of the art across a wide range of V+L benchmarks. However, little is known about the inner mechanisms that destine their impressive success.

Proposed method:  To reveal the secrets behind the scene, we present Value (Vision And Language Understanding Evaluation), a set of meticulously designed probing tasks (e.g., Visual Coreference Resolution, Visual Relation Detection) generalizable to standard pre-trained V+L models, to decipher the inner workings of multimodal pre-training (e.g., implicit knowledge garnered in individual attention heads, inherent cross-modal alignment learned through contextualized multimodal embeddings).

Value consists of a set of well-designed probing tasks that unveil the inner mechanisms of V+L pre-trained models across: (i) Multimodal Fusion Degree; (ii) Modality Importance; (iii) Cross-modal Interaction via probing visual coreferences; (iv) Image-to-image Interaction via probing visual relations; and (v) Text-to-text Interaction via probing learned linguistic knowledge.

Result: : (i) Pre-trained models exhibit a propensity for attending over text rather than images during inference. (ii) There exists a subset of attention heads that are tailored for capturing cross-modal interactions. (iii) Learned attention matrix in pre-trained models demonstrates patterns coherent with the latent alignment between image regions and textual words. (iv) Plotted attention patterns reveal visually-interpretable relations among image regions. (v) Pure linguistic knowledge is also effectively encoded in the attention heads.

## Spatiotemporal Attacks for Embodied Agents

Existing problems: Existing work on adversarial attacks have mainly focused on static scenes; however, it remains unclear whether such attacks are effective against embodied agents, which could navigate and interact with a dynamic environment. 

Proposed work:  In this work, we take the first step to study adversarial attacks for embodied agents.

Approach: In particular, we generate spatiotemporal perturbations to form 3D adversarial examples, which exploit the interaction history in both the temporal and spatial dimensions. Regarding the temporal dimension, since agents make predictions based on historical observations, we develop a trajectory attention module to explore scene view contributions, which further help localize 3D objects appeared with highest stimuli. By conciliating with clues from the temporal dimension, along the spatial dimension, we adversarially perturb the physical properties (e.g., texture and 3D shape) of the contextual objects that appeared in the most important scene views.

Result: Extensive experiments on the EQA-v1 dataset for several embodied tasks in both the white-box and black-box settings have been conducted, which demonstrate that
our perturbations have strong attack and generalization abilities. We also provide a discussion of adversarial training using our generated attacks, and a perceptual study indicating that contrary to the human vision system, current embodied agents are mostly more sensitive to object textures rather than shapes, which sheds some light on bridging the gap between human perception and embodied perception.

## A Cordial Sync: Going Beyond Marginal Policies for Multi-Agent Embodied Tasks

Existing problem: While multi-agent collaboration research has flourished in gridworld-like environments, relatively little work has considered visually rich domains.

Proposed task: Addressing this, we introduce the novel task FurnMove in which agents work together to move a piece of furniture through a living room to a goal. Unlike existing tasks, FurnMove requires agents to coordinate at every timestep.

Approach: We identify two challenges when training agents to complete FurnMove: existing decentralized action sampling procedures do not permit expressive joint action policies and, in tasks requiring close coordination, the number of failed actions dominates successful actions. To confront these challenges we introduce SYNC-policies (synchronize your actions coherently) and CORDIAL (coordination loss).

Result: Using SYNC-policies and CORDIAL, our agents achieve a 58% completion rate on FurnMove, an impressive absolute gain of 25 percentage points over competitive decentralized baselines.

## SoundSpaces: Audio-Visual Navigation in 3D Environments

Existing problems: Moving around in the world is naturally a multisensory experience, but today’s embodied agents are deaf—restricted to solely their visual perception of the environment.

Proposed task:  We introduce audio-visual navigation for complex, acoustically and visually realistic 3D environments. By both seeing and hearing, the agent must learn to navigate to a sounding object. The agent to (1) discover elements of the geometry of the physical space indicated by the reverberating audio and (2) detect and follow sound-emitting targets. We consider two variants of the navigation task: (1) AudioGoal, where the target is indicated by the sound it emits, and (2) AudioPointGoal, where the agent is additionally directed towards the goal location at the onset.

Approach:  We propose a multi-modal deep reinforcement learning approach to train navigation policies end-to-end from a stream of egocentric audio-visual observations.

Dataset:  SoundSpaces: a first-of-its-kind dataset of audio renderings based on geometrical acoustic simulations for two sets of publicly available 3D environments (Matterport3D and Replica), and we instrument Habitat to support the new sensor, making it possible to insert arbitrary sound sources in an array of real-world scanned environments.

Result: Our results show that audio greatly benefits embodied visual navigation in 3D spaces, and our work lays groundwork for new research in embodied AI with audio-visual perception. 

## Learning Object Relation Graph and Tentative Policy for Visual Navigation

Target: In this task, it is critical to learn informative visual representation and robust navigation policy.

Proposed method: Aiming to improve these two components, this paper proposes three complementary techniques, object relation graph (ORG), trial-driven imitation learning (IL), and a memory-augmented tentative policy network (TPN).

Approach details:  ORG improves visual representation learning by integrating object relationships, including category closeness and spatial correlations, e.g., a TV usually co-occurs with a remote spatially. Both Trial-driven IL and TPN underlie robust navigation policy, instructing the agent to escape from deadlock states, such as looping or
being stuck. Specifically, trial-driven IL is a type of supervision used in policy network training, while TPN, mimicking the IL supervision in unseen environment, is applied in testing.

Result: Experiment in the artificial environment AI2-Thor validates that each of the techniques is effective. When combined, the techniques bring significantly improvement over baseline methods in navigation effectiveness and efficiency in unseen environments. We report 22.8% and 23.5% increase in success rate and Success weighted by Path Length (SPL), respectively. Furthermore, our proposed trial-driven imitation learning empowers our agent to escape from deadlock states in training, while our tentative policy network allows our navigation system to leave deadlock states in unseen testing environments, thus further promoting navigation effectiveness and achieving better navigation performance.

## Occupancy Anticipation for Efficient Exploration and Navigation

Previous problems: State-of-the-art navigation methods leverage a spatial memory to generalize to new environments, but their occupancy maps are limited to capturing the geometric structures directly observed by the agent.

Proposed method: We propose occupancy anticipation, where the agent uses its egocentric RGB-D observations to infer the occupancy state beyond the visible regions.  In doing so, the agent builds its spatial awareness more rapidly, which facilitates efficient exploration and navigation in 3D environments.

Result: By exploiting context in both the egocentric views and top-down maps our model successfully anticipates a broader map of the environment, with performance significantly better than strong baselines.  Furthermore, when deployed for the sequential decision-making tasks of exploration and navigation, our model outperforms state-of-the-art methods on the Gibson and Matterport3D datasets. Our approach is the winning entry in the 2020 Habitat PointNav Challenge.

## PhraseClick: Toward Achieving Flexible Interactive Segmentation by Phrase and Click

Existing problem: Existing interactive object segmentation methods mainly take spatial interactions such as bounding boxes or clicks as input. However, these interactions do not contain information about explicit attributes of the target-of-interest and thus cannot quickly specify what the selected object exactly is, especially when there are diverse scales of candidate objects or the target-of-interest contains multiple objects. 

Proposed work:  We propose to employ phrase expressions as another interaction input to infer the attributes of target object. In this way, we can 1) leverage spatial clicks to locate the target object and 2) utilize semantic phrases to qualify the attributes of the target object. Specifically, the phrase expressions focus on “what” the target object is and the spatial clicks are in charge of “where” the target object is, which together help to accurately segment the target-of-interest with smaller number of interactions.

Result: Moreover, the proposed approach is flexible in terms of interaction modes and can efficiently handle complex scenarios by leveraging the strengths of each type of input. Our multimodal phrase+click approach achieves new state-of-the-art performance on interactive segmentation. To the best of our knowledge, this is the first work to leverage both clicks and phrases for interactive segmentation.

## Contrastive Learning for Weakly Supervised Phrase Grounding

Proposed method:  We show that phrase grounding can be learned by optimizing word-region attention to maximize a lower bound on mutual information between images and caption words. 

Finding: We formulate the problem as that of estimating mutual information between image regions and caption words. We demonstrate that maximizing a lower bound on mutual information with respect to parameters of a region-word attention mechanism results in learning to ground words in images. We also show that language models can be used to generate contextpreserving negative captions which greatly improve learning in comparison to randomly sampling negatives from training data.

Approach: Given pairs of images and captions, we maximize compatibility of the attention-weighted regions and the words in the corresponding caption, compared to non- corresponding pairs of images and captions. A key idea is to construct effective negative captions for learning through language model guided word substitutions.

Result: Training with our negatives yields a ∼ 10% absolute gain in accuracyover randomly-sampled negatives from the training data. Our weakly supervised phrase grounding model trained on COCO-Captions shows a healthy gain of 5.7% to achieve 76.7% accuracy on Flickr30K Entities benchmark.

## Visual Question Answering on Image Sets

Proposed work: . We introduce the task of Image-Set Visual Question Answering (ISVQA), which generalizes the commonly studied single-image VQA problem to multi-image settings. Taking a natural language question and a set of images as input, it aims to answer the question based on the content of the images. The questions can be about objects and relationships in one or more images or about the entire scene depicted by the image set.

Dataset: We introduce two ISVQA datasets – indoor and outdoor scenes. They simulate the real-world scenarios of indoor image collections and multiple car-mounted cameras, respectively. The indoor-scene dataset contains 91,479 human-annotated questions for 48,138 image sets, and the outdoor-scene dataset has 49,617 questions for 12,746 image sets. 

## VQA-LOL: Visual Question Answering under the Lens of Logic

Previous problem:  State-of-the-art models answer questions from the VQA dataset correctly, but struggle when asked a logical composition including negation, conjunction, disjunction, and antonyms. In this paper, we investigate whether visual question answering (VQA) systems trained to answer a question about an image, are able to answer the logical composition of multiple such questions. 

Proposed work:  We develop a model that improves on this metric substantially, while retaining VQA performance.

Dataset: We construct an augmentation of the VQA dataset as a benchmark, with questions containing logical compositions and linguistic transformations (negation, disjunction,
conjunction, and antonyms).

Approach: We propose our Lens of Logic (LOL) model which uses question-attention and logic-attention to understand logical connectives in the question, and a novel Frchet-Compatibility Loss, which ensures that the answers of the component questions and the composed question are consistent with the inferred logical operation.

Findings: When put under this Lens of Logic, state-of-the-art VQA models have difficulty in correctly answering these logically composed questions.

Results: Our model shows substantial improvement in learning logical compositions while retaining performance on VQA. We suggest this work as a move towards robustness by embedding logical connectives in visual understanding.


## TRRNet: Tiered Relation Reasoning for Compositional Visual Question Answering

Intention: Compositional visual question answering requires reasoning over both semantic and geometry object relations. 

Method details: We propose a novel tiered attention network for relation reasoning. The TRR network consists of a series of TRR units. Each TRR unit can be decomposed into four basic components: a root attention to model object level importance, a root to leaf attention passing module to select candidate objects based on root attention and generate pairwise relations, a leaf attention to model relation level importance and finally a message passing module for information communication between reasoning units.

Proposed method: We propose a novel tiered reasoning method that dynamically selects object level candidates based on language representations and generates robust pairwise relations within the selected candidate objects. Moreover, we propose a policy network that decides the appropriate reasoning steps based on question complexity and current reasoning status.

Result: The proposed tiered relation reasoning method can be compatible with the majority of the existing visual reasoning frameworks, leading to significant performance improvement with very little extra computational cost. In experiments, We achieve state-of-the-art performance on GQA dataset and competitive results on CLEVR datasets and VQAv2 dataset without functional program supervision.

## Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering

Previous problems: On the other hand, the data augmentation, as one of the major tricks for DNN, has been widely used in many computer vision tasks. However, there are few works studying the data augmentation problem for VQA and none of the existing image based augmentation schemes (such as rotation and flipping) can be directly applied to VQA due to its semantic structure – an himage, question, answeri triplet needs to be maintained correctly. 

Proposed method: In this paper, instead of directly manipulating images and questions, we use generated adversarial examples for both images and questions as the augmented data. The augmented examples do not change the visual properties presented in the image as well as the semantic meaning of the question, the correctness of the image, question, answer is thus still maintained.

Result: We then use adversarial learning to train a classic VQA model (BUTD) with our augmented data. We find that we not only improve the overall performance on VQAv2, but also can withstand adversarial attack effectively, compared to the baseline model. We show that the model trained with our method achieves 65.16% accuracy on the clean validation set, beating its vanilla training counterpart by 1.84%. Moreover, the adversarially trained model significantly increases accuracy on adversarial examples by 21.55%.

## RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving

Proposed work: In this work, we propose an efficient and accurate monocular 3D detection framework in single shot.

Previous methods: Most successful 3D detectors take the projection constraint from the 3D bounding box to the 2D box as an important component. Four edges of a 2D box provide only four constraints and the performance deteriorates dramatically with the small error of the 2D detector. 

Approach: Different from these approaches, our method predicts the nine perspective keypoints of a 3D bounding box in image space, and then utilize the geometric relationship of 3D and 2D perspectives to recover the dimension, location, and orientation in 3D space. 

Result: In this method, the properties of the object can be predicted stably even when the estimation of keypoints is very noisy, which enables us to obtain fast detection speed with a small architecture.  Training our method only uses the 3D properties of the object without any extra annotations, category-specific 3D shape priors, or depth maps. Our method is the first real-time system (FPS>24) for monocular image 3D detection while achieves state-of-the-art performance on the KITTI benchmark.

## Captioning Images Taken by People Who Are Blind


## Learning to Generate Grounded Visual Captions without Localization Supervision


## Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents


## Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline


## Learning Predictive Models from Observation and Interaction
























