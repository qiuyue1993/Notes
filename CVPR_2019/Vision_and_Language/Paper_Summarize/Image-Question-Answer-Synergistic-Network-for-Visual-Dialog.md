## Image-Question-Answer Synergistic Network for Visual Dialog

### Indexing:
- [Introduction](#Introduction)
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
### References
- [Image-Question-Answer Synergistic Network for Visual Dialog](https://arxiv.org/pdf/1902.09774.pdf)
---
