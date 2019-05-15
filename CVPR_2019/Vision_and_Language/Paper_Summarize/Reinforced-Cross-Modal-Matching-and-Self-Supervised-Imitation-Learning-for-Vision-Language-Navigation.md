## Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Reinforced Cross Modal Matching](#Reinforced-Cross-Modal-Matching)
- [Self-Supervised Imitation Learning](#Self-Supervised-Imitation-Learning)
- [References](#References)

---
### Introduction
**Abstract**
- We present two novel approaches, RCM and SIL, which combine the strength of reinforcement learning and self-supervised imitation learning for the vision-language navigation task.

**critical challenges**
- Cross-modal grounding
- Ill-posed feedback
- Generalization problems

**Contributions**
- Proposed a novel reinforced cross-modal matching (**RCM**) that enforces cross-modal grounding both locally and globally via reinforcement learning
- A **matching critic** is used to provide an intrinsic reward to encourage **global matching between instructions and trajectories**
- We further introduce a self-supervised imitation learning (**SIL**) method to explore unseen environments by imitating its own past, good decisions.
- Performs the new state-of-the-art performance on the Room-to-Room dataset.

---
### Related Work
#### Vision-and-Language Grounding
- VQA and Visual Dialog task focus on passive visual perception and the visual inputs are usually fixed.
- VLN solve the **dynamic multi-modal** grounding problem in **both temporal and spatial** spaces

#### Embodied Navigation Agent
- We are the first to propose to explore unseen environments for the VLN task.

#### Exploration
- We adapt SIL and validate its effectiveness and efficiency on the VLN.

---
### Reinforced Cross Modal Matching
#### Overview
- RCM framework mainly consists of two modules: a reasoning navigator, a matching critic.
- Intoduce two reward functions: an **extrinsic reward provided by the environment** to measure the success signal and the navigation error of each action; an **intrinsic reward** comes from **matching critic** to measure the alignment between the language instruction and navigator's trajectory.

#### Model
**Cross-Modal Reasoning Navigator**
- The navigator is a **policy-based agent** that **maps the input instruction** onto a sequence of **actions**.
- We design a cross-modal reasoning navigator that learns the **trajectory history**, the **focus of the textual instruction**, and the **local visual attention** which forms a cross-modal reasoning path to encourage the **local dynamics of both modalities**

*History Context*
- The history of the visual scene (trajectory) is encoded as a history context vector by **attention-based trajectory encoder LSTM**

*Visually Conditioned Textual Context*
- We learn the textual context conditioned on the history context.

*Textually Conditioned Visual Context*
- We compute the visual context based on the textual context.

*Action Prediction*
- Our action predictor considers the history context, the textual context, and the visual context, and decides which direction to go next based on them.
- It calculates the probability of each navigable direction using a bilinear dot product.

**Cross-Modal Matching Critic**
- This intrinsic reward encourages the **global matching** between the **language instruction** and the **navigator trajectory**.
- One way to realize this goal is to **measure the cycle-reconstruction reward**.
- We adopt an **attention-based sequence-to-sequence language model**
- The matching critic is pre-trained with human demonstrations via **supervised learning**

#### Learning
**Warm starting**
- In order to quickly approximate a relatively good policy, we use the **demonstration actions** to conduct **supervised learning with maximum likelihood estimation (MLE)**
- To learn a better and more generalizable policy, we then **switch to reinforcement learning** and introduce the extrinsic and intrinsic reward functions to refine the policy.

**Extrinsic Reward**
- We consider two metrics for the reward design.
- First metric: relative navigation distance, this indicates the **reduced distance** to the target location after taking action.
- Second metric: measures if the agent reaches a point within a threshold measured by the **distance from the target**
- To incorporate the influence of the action on the future and for the local greedy search, we use the **discounted cumulative reward**

**Intrinsic Reward**
- This reward encourages the agent to respect the instruction and penalizes the paths that deviate from what the instruction indicates.

---
### Self-Supervised Imitation Learning
- In this setting the agent is allowed to **explore unseen environments without ground-truth demonstrations**.
- It faciliates **lifelong learning** and **adaption to new environments**.

#### Details
- Given a natural language instruction, the navigator produces a set of possible trajectories and then stores the best trajectory into a replay buffer.
- The loss can also be interpreted as the supervised learning loss.

---
### References
- [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://arxiv.org/pdf/1811.10092.pdf)
---
