## Language-Driven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model

### Indexing:
- [Introduction](#Introduction)
- [Experiment and Result](#Experiment-and-Result)
- [References](#References)

---
### Introduction
- This paper focuses on a problem of **localizing an activity via a sentence query**

**Problems**
- Time-consumming due to the **dense frame-processing manner**
- Directly matching sentences with video content performs poorly due to the **large visual-semantic dicrepancy**

**Methods**
- A **recurrent** nerual network based **reinforcement learning** model which **selectively observes a sequence of frames** and **associates the given sentence with video content** in a **matching-based** manner.
- Extend the method to a **semantic matching renforcement learning** (SM-RL) model by **extracting semantic concepts of videos** and then **fusing them with global context features**.

---
### Experiment and Result
**Benchmark datasets**
- TACoS

- Charades-STA

- DiDeMo

**Results**
- state-of-the-art performance
- high detection speed
- both effectiveness and efficiency

---
### References
- [Language-Driven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Language-Driven_Temporal_Activity_Localization_A_Semantic_Matching_Reinforcement_Learning_Model_CVPR_2019_paper.pdf)
---
