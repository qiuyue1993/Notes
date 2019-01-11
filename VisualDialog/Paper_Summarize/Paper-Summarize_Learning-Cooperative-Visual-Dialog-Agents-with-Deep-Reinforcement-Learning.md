## Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning

### Indexing
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Cooperative Image Guessing Game](#Cooperative-Image-Guessing-Game)
- [Reinforcement Learning for Dialog Agents](#Reinforcement-Learning-for-Dialog-Agents)
- [Emergence of Grounded Dialog](#Emergence-of-Grounded-Dialog)
- [Experiments](#Experiments)
- [Conclusions](#Conclusions)
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
B-bot:
- Must provide a 'description' $\hat{y}$ of the unknown image *I* based only on the dialog history
Both players:
- Receive a reward from the environment inversely proportional to the error in this description under some metric $l(\hat{y}), y^{gt}$

**This is a general setting where the 'description' $\hat{y}$ can take on varying levels of specificity - from image embeddings to textual descriptions to pixel-level image generations**

#### Specific Instantiation
- Q-bot is tasked with estimating a vector embedding of image *I*
- Reward/error can be measured by simple Euclidean distance

---
### Reinforcement Learning for Dialog Agents

---
### Emergence of Grounded Dialog


---
### Experiments

---
### Conclusions

---
### References
- [Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1703.06585.pdf)

--




