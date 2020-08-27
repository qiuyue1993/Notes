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




## SoundSpaces: Audio-Visual Navigation in 3D Environments




## Learning Object Relation Graph and Tentative Policy for Visual Navigation












