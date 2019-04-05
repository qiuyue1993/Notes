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
- Image representation: a set of vectors, a vector corresponds to an **object** detected in the picture; **Bounding boxes**;
- Question representation: sentence embedding using gated recurrent unit network.

#### MuRel cell
- Visual Representation: a bag of N visual features, along with their bounding box coordinates.
- Bilinear fusion module: merges question and region feature vectors to provide a local multimodal embedding.
- Pairwise modeling component: update each multimodal representation with respect to its own spatial and visual context.

**Multimodal fusion**:
- Mutan: based on the Tucker decomposition of third-order tensors. This bilinear fusion model learns to focus on the relevant correlations between input dimensions. It models rich and fine-grained multimodal interactions, while keeping a relatively low number of parameters.
- In the MuRel cell, the local multimodal information is represented within a richer vectorial form $m_i$ which can encode more complex correlations between both modalities.

**Pairwise interactions**:
- Reason over multiple object that interact together.
- Each representation to be aware of the spatial and semantic context.
- Use pairwise relationship modeling where each region receives a message based on its relations to its neighbors.

- Compute a context vector for every region consisting in an aggregation of all the pairwise links coming into it.
- Use max operator in the agggregation function.
- Use bilinear fusion function for semantic and spatial information pairwise fusion.
- The MuREL cell's output is computed as a residual function.

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
