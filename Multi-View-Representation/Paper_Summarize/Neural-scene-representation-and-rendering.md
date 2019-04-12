## Neural scene representation and rendering

### Indexing
- [Introduction](#Introduction)
- [Related work](#Related-work)
- [Generative Query Network](Generative-Query-Network)
- [References](#References)
---
### Introduction
**Overview**
- GQN (Generative Query Network): Given images taken from multiple viewpoint of a 3D map and a query viewpoint, output the image from that
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
- Use VQE, one can transform a image from a viewpoint to a latent variant $z$; but it's difficult to operate $z$ to obtain image of desired
viewpoint.

### Conditional VAE
- Generate image conditioned by input condition $y$

---
### Generative Query Network
#### Comparison with CVAE
- Prior/Generation: use same structure for prior and generation
- Scene presentation from multiple viewpoint images.
- Recurrent Module: ConvLSTM

---
### References
- [Neural scene representation and rendering](https://science.sciencemag.org/content/sci/360/6394/1204.full.pdf)
- [Summarize Blog 1](https://www.slideshare.net/MasayaKaneko/neural-scene-representation-and-rendering-33d)
- [Summarize Blog 2](https://www.slideshare.net/DeepLearningJP2016/dlgqn-111725780)
---
