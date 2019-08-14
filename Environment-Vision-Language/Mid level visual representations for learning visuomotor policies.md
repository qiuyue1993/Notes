# Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies

## Indexing:
- [Introduction](#Introduction)
- [Approach](#Approach)
- [Experiments](#Experiments)
- [Comments](#Comments)
- [References](#References)
---
## Introduction
**Abstract**
- Discuss how much does **having visual priors abouth the world** assist in learning to perform **downstream motor tasks**
- By **integrating a generic perceptual skill set** within a **reinforcement learning framework**

**Finding and Approach**
- Finding: The correct choice of feature depends on the downstream task.
- Approach: introducing a **principled method for selecting a general-purpose feature set**
- Results: higher final performance with at least an order of magnitude less data than learning from scratch

**Simplifying assumptions**
- Selected **Locomotive Tasks** as the active tasks, and discussed the utility of mid-level vision on it
- Limitations of existing RL methods: difficulties in **long-range exploration** and **credit assignment with sparse rewards**
- Relaxing the **fixed set of mid-level features constraint** may improve the performance
- **Lifelong learning** of updating the visual estimators are important future research questions.

**Problems of existing RL**
- Previous research on RL of **pixel-to-torque** raised a question that **if all one needs from images can be learned from scratch using raw pixels by RL?**
- Requires massive amounts of data, resulting policies exhibit difficulties reproducing across environments with even modest visual differences

**Proposals**
- Including appropriate perceptual priors can alleviate these two phenomena, improving generalization and sample 
- We study how much **standard mid-level vision tasks and their associated features** can be used with **RL frameworks** in order to **train effective visuomotor policies**
- Analysis three questions: learning speed; generalization to unseen test spaces; whether a fixed feature could suffice or a set of features is required for supporting arbitrary motor tasks.
- We put forth a simple and practicle solver that takes a large set of features and outputs a smaller feature subset that minimizes the worst-case distance between the selected subset and the best-possible choice

**Findings**
- Appropriate **choice of feature** depends fiercely on the final task. Solving multiple active tasks therefore requires a set of features.

---
## Approach
### Overall
- Our setup assumes access to a set of features, where each feature is a function that can be applied to raw sensory data

### Using Mid-Level Vision for Active Tasks
- Choose 20 mid-level visual task 
- Freeze each encoder's weights to transform each observed image
- During training, only the agent policy is updated

### Core Questions
- **Sample Efficiency**: examine if an agent can learn faster
- **Generalization**: can agent generalize better to unseen spaces
- **Single Feature or Feature Set**: can a single feature support all downstream tasks? Is a set of features requried for gaining the feature benefits on arbitrary active tasks?
- **Findings**: depth estimation features perform well for visual exploration and object classification for target-driven navigation

### A Covering Set for Mid-Level Perception
- A compact feature set is desirable; therefore, we propose a **Max-Coverage Feature Selector** that curates a compact subset of features
- **With a measure of distance between features, we can explicitly minimize the worst-case distance between the best feature and out selected subset**; The taxonomy method defines exactly such a distance.

---
## Experiments



---
## Comments
- Does the mid-level perception contain "the fact that the world is 3D"?
- Is this no problems ? "realizing these gains requires careful selection of the mid-level perceptual skills". Curious about the explaination

---

## References
- [Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies](http://perceptual.actor/assets/main_paper.pdf)
