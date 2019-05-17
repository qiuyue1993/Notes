## Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation

### Indexing:
- [Introduction](#Introduction)
- [Method](#Method)
- [Experiments](#Experiments)
- [Analysis](#Analysis)
- [Related Work](#Related-Work)
- [Conclusion](#Conclusion)
- [References](#References)

---
### Introduction
**Abstract**
- We present the **Frontier Aware Search with backTracking (FAST) Navigator**, a general framework for action decoding.
- We use **asynchronous search to boost any VLN navigator by enabling explicit backtrack when an agent detects if it is lost**

**Compare with former works**
- Current approaches **make local action decisions**
- Or score entire trajectories using beam search
- We **balance local and global signals** when exploring an unobserved environment; This lets us act greedily but use global signals to backtrack when necessary.

**Problems of former works**
- Exposure bias: model **cannot perform accurately given its self-generated sequence** when trained on ground-truth data
- Deficiency of student forcing models: once the agent has **deviated from the correct path**, the original instruction no longer applies, which causes the confused agent to **fall into loops** sometimes.
- Beam search, which collects multiple global trajectories to score, **runs counter to** the goal of building an **efficent** agent.

**Solutions**
- Our FAST model is a form a **asynchronous search**, which **combines global and local knowledge to score** and **compare partial trajectories of different lengths**.
- We evaluate our progress by modeling how closely our previous actions align with the given text instructions, the result determines **which local action to take** and **whether the agent should backtrack**

---
### Method
#### Settings
- Input: instruction $X$, contains sentences describing a desired trajectory.
- Input at each step $t$: surrounding $V_t^{k}$, $k$ denotes the viewponit number
- Action: $a_1, a_2, ...,a_T \in A$， use panoramic action space, including a *stop* action

#### Learning Signals



#### Framework



#### Algorithm



---
### Experiments


---
### Analysis


---
### Related Work


---
### Conclusion


---
### References
- [Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation](https://arxiv.org/pdf/1903.02547.pdf)
- [Video](https://www.youtube.com/watch?v=AD9TNohXoPA&feature=youtu.be)
- [Code](https://github.com/Kelym/FAST)
---
