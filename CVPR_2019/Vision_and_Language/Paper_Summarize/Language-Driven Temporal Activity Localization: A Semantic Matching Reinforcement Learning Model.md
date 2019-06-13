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
- Current studies just focus on a limited set of actions **described at word level**.
- Similiar studies utilize sliding windows to generate dense proposals and every frame needs to be processed, which is **time-consuming**; Former works use **average pooling** to generate video features, thus temporal information might not be fully exploited.

**Methods**
- A **recurrent** nerual network based **reinforcement learning** model which **selectively observes a sequence of frames** and **associates the given sentence with video content** in a **matching-based** manner.
- Extend the method to a **semantic matching renforcement learning** (SM-RL) model by **extracting semantic concepts of videos** and then **fusing them with global context features**.

**Settings**
- Given the guidance of sentences
- The proposed model acts as a recurrent nn based agent 
- Which dynamically observes a sequence of video frames
- Finally outputs the temporal boundaries of the given sentence uery
- In particular, we introduce a state value, which measures the similarity of given sentence query and current observed video frame.

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
