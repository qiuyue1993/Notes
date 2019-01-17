## Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning

### Indexing
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Cooperative Image Guessing Game](#Cooperative-Image-Guessing-Game)
- [Reinforcement Learning for Dialog Agents](#Reinforcement-Learning-for-Dialog-Agents)
- [Emergence of Grounded Dialog](#Emergence-of-Grounded-Dialog)
- [Experiments](#Experiments)
- [Conclusions](#Conclusions)
- [Thoughts](#Thoughts)
- [References](#References)
---
### Introduction
<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/papersummarize_q-bot-a-bot_overview.png" width="400" hegiht="400" align=center/>

#### Problems of Visual Dialog Task
- Treat dialog as static supervised learning problem, rather than an  interactive agent learning problem than it naturally is.

#### Task
-  We pose a cooperative ‘image guessing’ game between two agents – Q-BOT and A-BOT–who communicate in natural language dialog so that Q-BOT can select an unseen image from a lineup of images. 
-  It is crucial for Q-bot and A-bot to learn to play to each other's strengths

##### Q-bot
- Shown a 1-sentence description (a caption) of an unseen image
- To communicate in natural language with the answering bot (A-bot)
- Objective: build a mental model of the unseen image purely from the natural language dialog, and then retrieve that image from a lineup of images.

##### A-bot
- Shown the image
- To communicate in natural language with the questions questioning bot (Q-bot)

#### Contributions
- Introduced a novel goal-driven training for visual question answering and dialog agents.
- Through a pure RL diagnostic task, they found that **Automatic emergence of grounded language and communication among 'visual' dialog agents with no human supervision**
- Through experiments on VisDial dataset, they found that RL fine-tuned bots significantly outperform the supervised bots.
- RL fine-tuned bots shifts strategies and asks question that the A-bot is better at answering.

---
### Related Work
#### Vision and Language
- Image Captioning
- [Visual Question Answering (VQA)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)
- [Visual Dialog](https://arxiv.org/abs/1611.08669)
- [AlphaGo](https://www.nature.com/articles/nature16961)

#### 20 Questions and Lewis Signaling Game
- Lewis Signaling (LS) game: cooperative game between two players - a *sender* and a *receiver*

#### Text-only or Classical Dialog
- [Li *et al.*](https://aclweb.org/anthology/D16-1127) proposed using RL for training dialog systems. However, they hand-define what a 'good' utterance/dialog looks like.

#### Emergence of Language
---
### Cooperative Image Guessing Game
#### Players and Roles
Involves two collaborative agents - a questioner bot (Q-bot) and an answer bot (A-bot) - **with an information asymmetry**
**Communication occurs for a fixed number of rounds**

Q-bot
- does not see image
- Primed with a 1-sentence description *c* of unseen image and asks 'questions'

A-bot
- sees an image *I*
- Answers questions

#### Game Objective in General
At each round, for
Q-bot:
- Must provide a 'description' $\hat{y}$ of the unknown image *I* based only on the dialog history
Both players:
- Receive a reward from the environment inversely proportional to the error in this description under some metric $l(\hat{y}), y^{gt}$

**This is a general setting where the 'description' $\hat{y}$ can take on varying levels of specificity - from image embeddings to textual descriptions to pixel-level image generations**

#### Specific Instantiation
- Q-bot is tasked with estimating a vector embedding of image *I*
- Reward/error can be measured by simple Euclidean distance

---
### Reinforcement Learning for Dialog Agents
#### Definition


#### Policy Networks for Q-bot and A-bot

#### Joint Training with Policy Gradients


---
### Emergence of Grounded Dialog
There are several challenging sub-tasks for Q-bot and A-bot to accomplish, such as:
- Learn a common language
- Develop mappings between symbols and image representations
- For A-bot, to learn to ground language in visual perception to answer questions
- For Q-bot, to predict plausible image representations

Therefore, they conduct a 'sanity check' on a synthetic dataset.

#### Setup
- Synthetic world with 64 unique images (with different shapes, colors, styles)
- Q-bot: to deduce two attributes of the image in order

#### Vocabulary
- A-bot: $V_A = \{1, 2, 3, 4\}$
- Q-bot: $V_Q = \{X, Y, Z\}$

#### Policy Learning
- Instantiate Q-bot and A-bot as fully specified tables of Q-values
- Apply tabular Q-learning with Monte Carlo estimation over $10k$ episodes to learn the policies
- *Couldn't understand*

#### Results
- The Q-bot and A-bot invented their own communication protocol.
- Essentially, they find the automatic emergence of grounded language and a communication protocol among ‘visual’ dialog agents without any human supervision!

---
### Experiments
#### Image Feature Regression
- Q-bot needs to regress to the vector embedding of image $I$ corresponding to the fc7 output from VGG16. 
- L2 distance metric is used in the reward computation

#### Training Strategies
Two training strategies are crucial to ensure or improve the convergence of RL framework.

##### Supervised Pretraining
- Conditioned on human dialog history, Q-bot is trained to generate the follow-up question by human1, A-bot is trained to generate the follow-up response by human2;  
- Both of the feature regression network and the CNN of A-bot is pretrained.

##### Curriculum Learning
- We continue supervised training for the first K (say 9) rounds of dialog and transition to policy-gradient updates for the remaining 10 − K rounds. We start at K = 9 and gradually anneal to 0.

Models are pretrained for 15 epochs on VisDial, after which we transition to policy-gradient training by annealing K down by 1 every epoch.
All LSTMs are 2-layered with 512-d hidden states.
We use Adam with a learning rate of $10^{-3}$, and clamp gradients to [−5, 5] to avoid explosion.

#### Model Ablations
Ablations experiments are aimed to have comparision between with/without RL, the importance of coordinated communication between Q-bot and A-bot.

- RL-full-QAf: full model (pretrained on VisDial and fine-tuned by RL)
- SL-pretrained: Purely supervised agents.
- Frozen-Q: Q-bot fixed by supervised pretrained initialization, A-bot to be trained
- Frozen-A: A-bot fixed by supervised pretrained initialization, Q-bot to be trained
- Frozen feature regression network

These models are evaluated along two dimensions: 
- How well they perform on the image guessing task
- How closely they emulate human dialogs

#### Evaluation: Guessing Game
- Image retrieval experiment based on the VisDial v0.5
- Communicate over 10 rounds
- Sort the entire test set in ascending distance to the predicted image and compute the rank of the source image.

Some observations from above result table:
- RL improves image identification
- All agents 'forget'; RL agents forget less
- RL leads to more informative dialog.

#### Evaluation: Emulating Human Dialog
- Evaluate A-bot on the retrieval metrics proposed in [Visual Dialog](https://arxiv.org/abs/1611.08669)
- Results show that the improvements on VisDial metrics are minor.
- Frozen-Q-multi (multi-task objective) performs best.

#### Human Study
- Evaluate whether humans can easily understand the Q-bot-A-bot dialog
- How image discriminative the interactions are.
- Results show that proposed methods successfully develop image-discriminative language and this language is interpretable.

---
### Conclusions
- Novel training framework for visually-grounded dialog agents by posing a cooperative 'image guessing' game between two agents.

---
### Thoughs
Problems of traditional Visual Dialog settings:
- Never steer conversation
- Poor evaluation metrics. 

Problems of Q-bot and A-bot settings:
- Emergence of language is somehow useless?
- Actually learnt to communicate with each other (Q-bot and A-bot), rather than communicate with human.

---
### References
- [Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1703.06585.pdf)

--




