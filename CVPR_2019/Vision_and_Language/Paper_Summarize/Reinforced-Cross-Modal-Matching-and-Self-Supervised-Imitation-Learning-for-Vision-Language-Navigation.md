## Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Reinforced Cross Modal Matching](#Reinforced-Cross-Modal-Matching)
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
- The navigator is a policy-based agent that maps the input instruction onto a sequence of actions.
- We 


**Cross-Modal Matching Critic**


#### Learning


---
### References
- [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://arxiv.org/pdf/1811.10092.pdf)
---
