## Two can play this Game: Visual Dialog with Discriminative Question Generation and Answering 

### Indexing
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Approach](#Approach)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
- [Thoughts](#Thoughts)
- [References](#References)

---
### Introduction
- Overview 

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_overview.png" width="400" hegiht="400" align=center/> 

Abstract
- In this paper, we demonstrate a **simple symmetric discriminative baseline**, that can be applied to **both predicting an answer** as well as **predicting a question**.

Contribution
- In this paper we develop a deep net architecture that predicts an answer given a question, a caption, an image, and a question-answer history.  The proposed approach **outperforms existing baselines**.
-  We re-purpose the visual dialog dataset and demonstrate that our developed architecture is **applicable to  question prediction setup** without signiﬁcant changes. 
-  To  **produce a visual dialog**, our discriminative questioning and answering modules communicate with each other.
---
### Related Work
#### Image Captioning
- Classical methods formulate image captioning as a retrieval problem.
- As constructing a database of captions that is sufficient for describing a reasonably large fraction of images seems prohibive, recurrent nerual nets (RNNs) decompose the space of a caption into a product space of individual words.

#### Visual Question Answering
- This task if often used as a testbed for reasoning capabilities of deep nets. 
- Using a variety of datasets, models based on multi-modal representation and attention, deep net architecture developments and dynamic memory nets have been discussed.
- **Despite these efforts, it is hard to assess the reasoning capabilities of present day deep nets and differentiate them from memorization of training set statistics.**

#### Visual Question Generation
-  In spirit similar to question answering but often involving a slightly more complex language part is the task of visual question generation.
-  It has been proposed very recently and is still very much an open-ended topic. 

#### Visual Dialog
- Visual Dialog  involves both generation of questions and corresponding answers.
- [Transferring knowledge from discriminative learning to a generative visual dialog model](https://arxiv.org/pdf/1706.01554.pdf) proposed a generator-discriminator architecture where the outputs of the generator are improved using a perceptual loss from a pre-trained discriminator. 

---
### Approach
- Brief difference between proposed method and traditional methods

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_structure-brief.png" width="400" hegiht="400" align=center/> 

- Overview of proposed architecture

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_proposed-structure.png" width="600" hegiht="400" align=center/> 


#### Overview
- It is the purpose of this paper to **maximally utilize the informativeness of options**, i.e., to use early option input. Hence, we focus on discriminative visual dialog systems. 
- In this paper, we also provide results for **question generation**.
- To this end we develop a **uniﬁed deep net architecture** for both visual question answering and question generation. 

Problem Definition:
- VisDial Dataset: tuples $$(I, C, H_t, Q_t, A_t)$$
- Generative techniques:  use embeddings of those three elements (Image, History and Caption, Question), or a combination thereof to **model a probability distribution over all possible answers**.
- Discriminative techniques: operate on a set of answers, particularly their embeddings, and assess the ﬁtness of every set member w.r.t. the remaining data, i.e., the image $I$, the history $H_t$, the caption $C$ and the question $Q_t$.

#### Unified Deep Net Architecture
- LSTM: current question embedding, caption, a set of answer options
- History Embedding: Question LSTM to embed question, Answer LSTM to embed answer, FC layer beyond them to compute a simple vector. Concatenate all histoyr vector to form the history embedding.
- Concatenate: all vector introduced above
- Similarity network: predict a probability distribution over the possibble anwers.

##### Question and Answer Embedding
- A *Stop* token is introduced to mark the end of question in VisDial dataset.
- LSTM

##### Caption Embedding
- LSTM

##### Image Representation
- Pretrained CNN Features

##### History Embedding
- LSTM for Question and Answer separately
- FC Layer to fusion question and answer vector
To tackle variable length history:
- Introduce an *Empty* token
- For missing question-answer rounds, pass (Empty, Stop) sequence to fore-mentioned architecture.

##### Similarity Scoring+Fusion Network
- Ensemble Representation: $L_S = L_Q + L_I + L_C + (T-1) * L_H + L_O$
- Perform similarity scoring and feature fusion jointly using MLP.

#### Network Training
- Adam optimizer with a learning rate of $10^{-3}$
- Sharing the weights of the language embedding greatly helps in learning better word representations.
- Two hidden layered MLP nets performs better than one layer.
- **All our models converge in under 5 epochs of training on VisDial dataset.**

#### Implementation Details
- **Utilize deep metric learning and a self-attention mechanism to train a discriminator network which leverages the availability of answer options.  We achieve this via a simple LSTM-MLP approach.**

#### VisDial-Q Dataset and Evaluation
-  We create a similar ‘VisDial-Q evaluation protocol.’ A visual question generation system is required to choose one out of 100 next question candidates for a given question-answer pair.
- Select 100 candidate follow-up questions to a given QA pairs as the union of *Correct*, *Plausible*, *Popular*, *Random*

---
### Experiments
#### Datasets
- Trained on VisDial v0.9 dataset.

#### Evaluation Metrics
- Image Captioning: BLEU, ROUGE, METEOR
- Visual Dialog: Recall@k, MRR, MR

#### Quantitative Accessment
##### Visual Question Answering
- VisDial evaluation metrics

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_VisDial-Evaluation.png" width="400" hegiht="400" align=center/> 

##### Visual Question Generation
- VisDial-Q evaluation metrics

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_VisDial-Q-Evaluation.png" width="400" hegiht="400" align=center/> 

Obtained intuitive insights
- Predicting the next question is a much more difficult task than answering a question without context.
- Image and history cues are much more important forthe question prediction task than for answer prediction.


#### Qualitative Evaluation
- Examples of generated dialogs

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_VD-with-Discriminative-Question-Generation-and-Answering_Generated-Dialogs.png" width="600" hegiht="400" align=center/> 

---
### Conclusion
- We developed a **discriminative method** for the visual dialog task, i.e., predicting an answer given question and context.
-  More importantly, our approach can be applied with almost no change to prediction of a question given context.
-  We introduce the **VisDial-Q evaluation protocol** to quantitatively assess this task and also illustrate how to combine both discriminative methods to obtain a system for visual dialog. 

---
### Thoughts
- Sometimes the order of the questions is non-sense?
- How much are information from images used?
 - How to evaluate the quality of dialogs?
 
---
### References
- [Two can play this Game: Visual Dialog with Discriminative Question Generation and Answering ](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jain_Two_Can_Play_CVPR_2018_paper.pdf)

---

