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

- Previous research on RL of **pixel-to-torque** raised a question that **if all one needs from images can be learned from scratch using raw pixels by RL?**

---
## Approach

---
## Experiments



---
## Comments
- Does the mid-level perception contain "the fact that the world is 3D"?
- Is this no problems ? "realizing these gains requires careful selection of the mid-level perceptual skills". Curious about the explaination

---

## References
- [Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies](http://perceptual.actor/assets/main_paper.pdf)
