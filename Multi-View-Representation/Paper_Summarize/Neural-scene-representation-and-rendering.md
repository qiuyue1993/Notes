## Neural scene representation and rendering

### Indexing
- [Introduction](#Introduction)
- [Related work](#Related-work)
- [Generative Query Network](#Generative-Query-Network)
- [Experiments](#Experiments)
- [References](#References)
---
### Introduction
**Overview**

<img src="https://github.com/qiuyue1993/Notes/blob/master/Multi-View-Representation/images/Paper-Summarize_Generative-Query-Network_Overall-framework.png" width="600" hegiht="400" align=center/>


- GQN (Generative Query Network): The GQN takes as input **images of a scene taken from different viewpoints**, constructs an **internal representation**, and uses this representation to **predict** the appearance of that scene from **previously unobserved viewpoints**.
viewpoint (render).
- Scene representation: Multiple viewpoint images -> scene representation
- Rendering image: scene representation, viewpoint -> image

**Comparison with SfM/SLAM**
- SfM/SLAM: 3D Map estimation from multiple images; Localization of given image; Rendering;
- GQN: 3D scene representation from multiple images; Rendering

**Merits of GQN**
- No pointcloud CNN
- Add operation of multiview fusion.

---
### Related work
#### VAE (Variational Autoencoder)
- Use VQE, one can transform a image from a viewpoint to a latent variant $z$; but it's difficult to operate $z$ to obtain image of desired viewpoint.

### Conditional VAE
- Generate image conditioned by input condition $y$

---
### Generative Query Network
#### Comparison with CVAE
- Prior/Generation: use same structure for prior and generation
- Scene presentation from multiple viewpoint images.
- Recurrent Module: ConvLSTM

### The Usefulness of Generative Query Network
- The GQN's generation network can 'imagine' previously unobserved scenes from new viewpoints with remarkable precision.
- The GQN's representation network can learn to count, localise and classify objects without any object-level labels.
- The GQN can represent, measure and reduce uncertainty. It can combine multiple partial views of a scene to from a coherent whole.
- The GQN's representation allows for robust, data-efficient reinforcement learning.
---
### Experiments


---
### References
- [Neural scene representation and rendering](https://science.sciencemag.org/content/sci/360/6394/1204.full.pdf)
- [Official Blog](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
- [Summarize Blog 1](https://www.slideshare.net/MasayaKaneko/neural-scene-representation-and-rendering-33d)
- [Summarize Blog 2](https://www.slideshare.net/DeepLearningJP2016/dlgqn-111725780)
---
