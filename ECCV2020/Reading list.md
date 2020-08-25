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

## 








