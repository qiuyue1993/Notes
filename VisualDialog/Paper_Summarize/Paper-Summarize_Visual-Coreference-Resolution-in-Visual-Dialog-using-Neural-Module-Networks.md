## Visual Coreference Resolution in Visual Dialog using Neural Module Networks

### Indexing
- [Introduction](#Introduction)
- [Approach](#Approach)
- [Experiments](#Experiments)
- [Conclusions](#Conclusions)
- [Thoughts](#Thoughts)
- [References](#References)
---
### Introduction
- Overview of proposed visual coreference resolution

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Coreference-Resolution-in-Visual-Dialog_overview.png" width="600" hegiht="200" align=center/> 

#### Problem Definition
-  We focus on one such problem called **visual coreference resolution** that involves determining which words, typically noun phrases and pronouns, co-refer to the same entity/object instance in an image.
-  In this work, we propose a **neural module network architecture** for visual dialog by introducing two novel modules—**Refer** and **Exclude**—that perform explicit, grounded, coreference resolution at a **ﬁner word level**.

#### Coreference Resolution
- Coreference resolution: The linguistic community deﬁnes coreference resolution as the task of **clustering phrases, such as noun phrases and pronouns, which refer to the same entity in the world**
- Visual Coreference resolution: links the coreferences to an entity in the visual data.

#### Contribution
- Neural module network architecture for visual dialog. Specifically, proposed two novel modules - Refer and Exclude - that perform explicit, grounded, coreference resolution in visual dialog.
-  We propose a novel way to handle captions using neural module networks at a word-level granularity ﬁner than a traditional sentence-level encoding. 
---
### Approach
- Overview of proposed model architecture

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Coreference-Resolution-in-Visual-Dialog_overview-of-proposed-model.png" width="600" hegiht="200" align=center/> 

As rounds of dialog progress, the model collects unique entities and their corresponding visual groundings, and uses this reference pool to resolve any coreferences in subsequent questions. 
Three broad componets:
- Program Generation: A *program* is predicted for the current question $Q_t$
- Program Execution: The predicted program is executed by dynamically connecting neural modules to produce a *context* vector summarizing the semantic information required to answer $Q_t$ for the context $(I, H)$
- Answer Decoding: The context vector $c_t$ is used to obtain the final answer $\hat{A_t}$

---
### Experiments
#### MNIST Dialog Dataset
##### Dataset
- Images: composed from $4 \times 4$ grid of MNIST digits.
- Digits: four attributs (digit class, color, stroke and background color)
- Dialog: 10 question-answer pairs.
- Questions: generated through language templates. Contains questions of query attributes of target digits, count digits with similar attribute, need tracking of target digits.

##### Results
- Proposed CorefNMN outperforms all other models with near perfect accuracy of 00.3%.

#### VisDial Dataset
- Results on VisDial v1.0 dataset

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Coreference-Resolution-in-Visual-Dialog_results-on-VisDial-v1.png" width="600" hegiht="300" align=center/> 

- Results show that proposed method outperforms all previous methods in VisDial Dataset.
---
### Conclusions
- Novel model for visual dialog based on neural module networks that provides an introspective reasoning about visual coreferences.
- Experiments on both the MNIST dialog dataset and VisDial dataset show that the proposed model is more interpretable, grounded, and consistent.
---
### Thoughts
- Can be further used for applications such as Visual Dialogs with pointing questions.
- As using the setting of visual dialog, the bot never steers the dialog.
---
### References
- [Visual Coreference Resolution in Visual Dialog using Neural Module Networks](https://arxiv.org/pdf/1809.01816.pdf)
---
