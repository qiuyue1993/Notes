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
- A MuRel cell **iteratively updates the region state** vectors, each time refining the representations with **contextual** and **question** information.
- The **weights are shared** across the cells, which enables compact parametrization and good generalization.
- At step $t=T$, the representations are aggregated with a global max pooling operation to provide a single vector; This scene representation contains information about **objects**, the **spatial** and **semantic** relations between them.
- The scene representation is merged with the question embedding $q$ to compute a score for every possible answer.

**Visualizing MuRel network**
- The proposed model can be used to **highlight important relations** between image regions for answering a specific question.
- The proposed model also can be used to visualize **the pairwise relationships** involved in the prediction of MuRel cell.

**Connection to previous work (comparison with the FiLM network)**
- Dataset: real VQA dataset vs. synthetic CLEVR dataset
- Residual cells: one per interation vs. multiple cells
- Multimodal interaction: bilinear fusion strategy vs. feature-wise affine modulation.
- Spatial structure of the image representation: non trivial relational modeling vs. locally-connected graph.

---
### Experiments
#### Experimental setup
- Datasets: VQA2.0; VQA Changing Priors v2; TDIUC (12 well-defined types of question; biggest dataset for VQA)
- Hyper-parameters: Bottom-up features to represent image as a set of 36 localized regions; Pretrained Skip-throught encoder of question embedding; Adam optimizer with a learning scheduler.

#### Model validation
- Comparison to Attention-based model (Mutan): Significant gain on three datasets.


- Ablation study: Ablation on **Pairwise module** and **Iterative Process**; Model with pairwise module and iterative process results in a best performance.


- Number of reasoning steps: Ablation of one, two, three, four layers of iterative MuRel cells. Model with three iterative layers results in the best overall performance, whilist the model with four iterative layers performs best in counting questions.

#### State of the art comparison
- VQA2.0: Obtain same level of overall accuracy with VQA Challenge 2018 champion while did not extensively tune the hyperparameters. 


- TDIUC: Obtain state-of-the-art results on Overall accuracy and the arithmetic mean of per-type accuracies (A-MPT). Lower harmonic mean of per-type accuracies (H-MPT) as the low score on the Utility and Affordances task which are not directly related to the visual understanding of the scene.


- VQA-CP v2: state-of-the-art overall accuracy meanwhile less prone to question-based overfitting than classical attention architectures.

#### Qualitative results
- Iterations through the MuRel cell tend to **gradually discard regions**, keeping only the most relevant ones.
- VQA models are often subject to linguistic bias, the MuRel network acctually relies on the visual information to answer questions.
---
### Conclusion
- Our system is based on **rich representations of visual image regions** that are **progressively merged** with the question representation.
- We also included **region relations with pairwise combinations**.

---
### References
- [MUREL: Multimodal Relational Reasoning for Visual Question Answering](https://arxiv.org/pdf/1902.09487.pdf)
- [Code](https://github.com/Cadene/murel.bootstrap.pytorch)
