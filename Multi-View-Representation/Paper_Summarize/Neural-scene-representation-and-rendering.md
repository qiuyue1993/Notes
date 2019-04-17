## Neural scene representation and rendering

### Indexing
- [Introduction](#Introduction)
- [Related work](#Related-work)
- [Generative Query Network](#Generative-Query-Network)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
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

### Compared to former work
- Classical neural approaches (autoencoding and density models) are required to capture **only the distribution of observed images** and there is **no explicit mechanism to encourage learning of how different views of the same 3D scene relate to one another**.
- Viewpoint transformation networks have far been **nonprobabilistic and limited in scale**.
- GQN can be **augmented** with a second "generator" that given an image of a scene, **predicts the viewpoint** from which it was taken, providing a **new source of gradients** with which to train the representation network. 

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
#### Rooms with multiple objects

**Scene Setting**
- Scenes in a square room containing a variety of objects.
- Wall textures, shapes, positions, colors of the object and lights are randomized.

**Training**
- The model observes only a small number of images (in this experiment, fewer than five) during training.

**Results**
- t-SNE visualization of GQN scene representation vectors shows clear **clustering of images of the same scene**, **despite** marked changes in **viewpoint**.
- GQN exhibits **compositional** behavior.
- Object color, shape, size, light position, and, to lesser extent, object positions are indeed **factorized**.
- GQN is able to carry out **scene algebra**, by adding and subtracting representations of related scenes, the object and scene properties can be controlled.
- GQN also learns to **integrate information from different viewpoints** in an efficient and consistent manner.

#### Control of a robotic arm


#### Partially observed maze environments


---
### Conclusion

---
### References
- [Neural scene representation and rendering](https://science.sciencemag.org/content/sci/360/6394/1204.full.pdf)
- [Official Blog](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
- [Summarize Blog 1](https://www.slideshare.net/MasayaKaneko/neural-scene-representation-and-rendering-33d)
- [Summarize Blog 2](https://www.slideshare.net/DeepLearningJP2016/dlgqn-111725780)
---
