## Image-Question-Answer Synergistic Network for Visual Dialog

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Synergistic Network](#Synergistic-Network)
- [Extension to the Generative Model](#Extension-to-the-Generative-Model)
- [Experiments](#Experiments)
- [References](#References)

---
### Introduction
**Abstract**
- Propose a **image-question-answer synergistic** network to value the **role of the answer** for precise visual dialog.
- Propose a **two-stage** solution:
- Stage 1: candidate answers are **coarsely scored** according to their relevance to the image and question pair.
- Stage 2: answers with high probability of being correct are **re-ranked** by synergizing with image and question.
- We also **improve the N-pair loss function to solve the class-imbalanced problem** in the discriminative model.
- Outperform the champion of Visual Dialog Challenge 2018 on NDCG (Normalized Discounted Cucumlative gain).

**Problem of former methods**
- The **scoring method is insufficient** to capture the similarity between inputs and answers, since the vecotr of inputs and answers have been separately learned without deep fusion.
- Both generative and discriminative models tend to give short and safe answer, as their fusion methods focus on the major signal in short answer but will not look into details in a longer one.

---
### Related Work
#### Visual Question Answering (VQA)
- Main categories of VQA: early fusion models; later fusion models; external knowledge-based models.

#### Visual Dialog
- Extending the single turn dialog task (VQA) to a multi-turn one.
**Visual dialog task**
- Visual grounding
- Visual Question Answering
- Image captioning

---
### Synergistic Network
#### Primary Stage
**Abstract**
- Learn **representative vectors of the image, history, and question** using a **co-attention module**
- **Calculate the score** of each candidate answer to **separate hard answers from easy ones**.

**Details**
- An **encoder-decoder** solution is adopted.
- Using **MFB** (Multi-modal factorized bilinear pooling) to learn the unified vector of the question and history.
- Attention on history and image are applied.
- **N-pair loss** to measure the error.
- **MFB is expected to provide a richer representation than other bilinear methods, such as MLB and MCB**

**Tasks**
- **De-reference** in the multi-turn conversations.
- **Locate** the objects in the image mentioned in the current question.

#### Synergistic Stage
**Abstract**
- Select hard answers together with their questions to **form question-answer pairs**.
- These pairs further **coordinate with the image and history to predict scores**.

**Details**
- Using paired question-answer, history vector to learn the image's attention parameters
- Fusing image, history, question-answer vectors with MFB.
- Classify the correct answer by softmax.

---
### Extension to the Generative Model
- The **generative model can also be used** to score the candidate answers and **seamlessly** works with the proposed image question-answer synergistic method.

---
### Experiments
#### Dataset and Evaluation Metric
- NDCG is **invariant to the order of options** with identical relevance.

#### Implementation Details
- The LSTMs of the question and history are two layered, while it is one layered for the answer in the primary stage and the question-answer pair in the synergistic stage.
- The hidden state dimension *d* for all LSTMs and CNN is 512.
- For bilinear pooling, we set *k* to 5 and *l* to 1000.

#### Comparison with the State-of-the-art
- Compared with: Later Fusion (LF), Hierarchical Recurrent Encoder (HRE), Memory Network (MN), MN-att, LF-att
- Achieve the highest NDCG on the test-standard server of Visual Dialog Challenge 2018.

#### Ablation Study
- The performance **improves while the imbalance impact parameter decays**, which makes the model pay more attention to incorrect answers scored near or higher than the correct answer. 
- Feeding more samples by increasing M improves performance to produce the best model at *M30*
- The **primary stage show more importance** to the synergistic stage.
- The primary stage is also necessary to balance the memory cost.
- In **generative model**, we **increase** the selected answer number **N** from 10 to 30 in the second stage.

#### Qualitative Analysis
- **One-stage model tend to give a safe answer** such as 'NO'
- Surprisingly, our model can sometimes even **generate better answers** than those provided.

---
### Conclusions
- Developed a synergistic network that **jointly learns the representation of the image, question, answer, and history in a single step**.
- **Imporove the N-pair loss function** to solve the **class-imbalanced problem** in the discriminative model.

---
### References
- [Image-Question-Answer Synergistic Network for Visual Dialog](https://arxiv.org/pdf/1902.09774.pdf)
---
