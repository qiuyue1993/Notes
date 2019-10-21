# Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations

## Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Formulation](#Formulation)
- [Experiments](#Experiments)
- [References](#References)
---
## Introduction
### Abstract
- We propose Scene Representation Networks (SRNs), a **continuous**, **3D-structure-aware** scene representation that encodes both **geometry and appearance**.
- We demonstate the potential of SRNs by evaluating them for **novel view synthesis**, **few-shot reconstruction**, **joint shape and appearance interpolation**, and **unsupervised discovery of a non-rigid face model**

### Limitations of previous scene representation
#### Classic 3D scene representations (voxel grids, point clouds, meshes)
- **Discrete**, limiting achievable spatial resolution and only **sparsely** sampling the underlying smoooth surfaces of a scene.

#### Neural scene representations
- Do **not explicitly** represent or reconstruct scene **geometry**.
- Black-box neural renderer
- Fail to capture 3D operations, such as camera translation or rotation.
- Lack guarantees on multi-view consistency of the rendered images.

### Key idea
- Representing a scene implicitly as a **continuous, differentiable function** that **maps a 3D world coordinate to a feature-based representation** of the scene properties at that coordinate.
- SRNs generate high-quality images **without any 2D convolutions**, exclusively operating on individual pixels, which enables **image generation at arbitrary resolutions**

### Limitations
- SRNs currently do not model view-dependent effects and reconstruct shape and appearance in an entangled manner. 

---
## Related-Work


---
## References
- [Paper](https://arxiv.org/pdf/1906.01618.pdf)
- [Code](https://github.com/vsitzmann/scene-representation-networks)
---
