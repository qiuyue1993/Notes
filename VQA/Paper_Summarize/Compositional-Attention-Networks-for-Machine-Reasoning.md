## Compositional Attention Networks for Machine Reasoning

### Indexing:
- [Introduction](#Introduction)
- [References](#References)
---
### Introduction
**Overview**
- Proposed MAC network, a fully differentiable neural network architecture, designed to facilitate **explicit and expressive reasoning**.
- Proposed model approaches problems by decomposing them into a series of **attention-based reasoning steps**.
- Each step performed by Memory Attention Composition (MAC) cell that maintains **control and memory**

**Problems of former Symbolic methods**
- Rely on **externally provided structured representations and functional programs**, brittle handcrafted parsers or expert demonstrations.
- Require relatively **complex** multi-stage reinforcement learning **training schemes**.

**Merits**
- State-of-the-art accuracy on CLEVR
- Computationally-efficient
- Data-efficient
- Can be adapted to novel situations and diverse language.
- The universal design of the MAC cell serves as a **structural prior** that encourages the network to solve problems by **decomposing them into
a sequence of attention-based reasoning operations** that are directly inferred from the data.

---
### Rerences
- [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf)

---
