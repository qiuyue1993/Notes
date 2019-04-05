## MUREL: Multimodal Relational Reasoning for Visual Question Answering

### Indexing:
- [Introduction](#Introduction)
- [Related work](#Related-work)
- [MuRel approach](#MuRel-approach)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
- [References](#References)

---
### Introduction
Problem of **Multimodal attentional networks**:
- Attention mechanism is arguably **insufficient** to model complex reasoning features required for VQA.

**MuRel**:
- End-to-end to **reason over real images**.
- MuRel cell: atomic reasoning primitive representing interactions between question and image regions.
- MuRel **progressively** refines visual and question interactions.

---
### Related work
#### Visual reasoning
**Explicit reasoning**
- A neural network reads the question and **generates a program**, corresponding to a graph of elementary neural operations that
process the image.

**Implicit reasoning**
- [FiLM](https://arxiv.org/pdf/1709.07871.pdf): modulates the visual feature map with an affine transformation whose parameters depend on the question.
- [Relation Network](https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning.pdf): reason over all the possible pairs of objects in the picture.

### VQA on real data
**Tensor decomposition frameworks**
- [Mutan](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ben-younes_MUTAN_Multimodal_Tucker_ICCV_2017_paper.pdf)
- [MCB](https://arxiv.org/pdf/1606.01847.pdf)
- [MLB](https://arxiv.org/pdf/1610.04325.pdf)

---
### MuRel approach
#### Overview
- Image representation: $\{V_i\}_(i)$

#### MuRel cell



#### MuRel network




---
### Experiments

---
### Conclusion
- Our system is based on **rich representations of visual image regions** that are **progressively merged** with the question representation.
- We also included **region relations with pairwise combinations**.

---
### References
- [MUREL: Multimodal Relational Reasoning for Visual Question Answering](https://arxiv.org/pdf/1902.09487.pdf)
- [Code](https://github.com/Cadene/murel.bootstrap.pytorch)
