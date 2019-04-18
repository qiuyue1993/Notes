## BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding

### Indexing:
- [Introduction](#Introduction)
- [Transformer](#Transformer)
- [BERT](#BERT)
- [References](#References)

---
### Introduction
**Core of BERT**
- Learning the **relationship between two sentences** by predicting if one of the sentences is the sequent sentence of the other.
- Delete some words in a sentence and train to **predict those words**.
- Use large scale **transformer model** and train the above two simultaneously by setting **two loss functions**.


**Pre-trained model**
- NLP tasks can benefit from Pre-training on large scale dataset. 

**Two important topics in NLP**
- How to fully exploit **unlabeled data**. **BERT** provides two direction to exploit unlabeled data.
- **Transformer** show strong power in machine translation and the larger dataset can reveal the advantages of this structure.

**Computation power**
- Although BERT is amazing, it requires a lot of computation power.

---
### Transformer
- Use attention mechanism and fully connect to process text.
- Self attention.
- Multi-Head Attention.
- Encoder-decoder framework.

---
### BERT
- Transformer
- Bi-directional: While processing a word, BERT will use the information of words before and after the words by cover words in a sentence randomly.

**Input Representation (Embedding)**
- Input sentence -> Token Embeddings + Segment Embeddings + Position Embeddings

**Pre-training Process**
- Multi-task 1: predict masked words (15%)
- Multi-task 2: binary classify nextsentence.
- Dataset: BooksCorpus (8 billion words), Wikipedia (25 billion words)

- Mask words: 80% use [MASK], 10% use other words, 10% use the former words.

**Fine-tuning Process**
- BERT can be applied to most NLP tasks by fine-tuning.

**Tips**
- BERT model have 12 layers and not wide (1,024)
- MLM (Masked Language Model): use the left and right words simultaneously
- BERT is unsupervised
- Deep learning is representation learning.
- Scale matters.
- Pre-training is important.

---
### References
- [BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Summarize blog 1](https://www.jiqizhixin.com/articles/2018-11-01-9)
- [Summarize blog 2](https://zhuanlan.zhihu.com/p/46652512)
- [Summarize blog for Transformer](https://zhuanlan.zhihu.com/p/44121378)
---
