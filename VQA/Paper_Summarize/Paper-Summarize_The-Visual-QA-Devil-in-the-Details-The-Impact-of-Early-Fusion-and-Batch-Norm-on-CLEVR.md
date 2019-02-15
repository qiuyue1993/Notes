## The Visual QA Devil in the Details: The Impact of Early Fusion and Batch Norm on CLEVR

### Indexing
- [Introduction](#Introduction)
- [Models](#Models)
- [Experiments and Conclusions](#Experiments-and-Conclusions)
- [References](#References)
---
### Introduction
- In particular, we ﬁnd that **early fusion of language and vision** provides large performance improvements. 
- We propose a simple module we call **Multimodal Core (MC)**, which we hypothesize performs the fundamental operations for multimodal tasks. 

---
### Models
- Simple Feed Forward Network



- Image CNN: 4 layers of Conv, ReLU, 128 kernels of size 3-by-3 and strides 2 and batch-norm
- Question LSTM: trans question to 128 dimension

We investigate four variants for fusing the vision and language.
- early + batch + SFF (performs best, called as **MC**)
- late + batch + SFF
- early + SFF
- late + SFF
---
### Experiments and Conclusions
- Experiments on CLEVR

- **Early fusion** is indeed critical for CLEVR;
- **Batch-norm** shows small but non-trivial gains, especially on harder questions like counting or comparing numbers.
- We see a drastic improvement of using MC relative to SAN, suggesting that SAN processes the multimodal features too late in the pipeline, and too "shallowly".

- Although late fusion may work on more biased Visual QA datasets, where exploiting language biases plays a more prominent role.

- So what is important for CLEVR performance? CLEVR high-performing models like RN and FiLM differ from earlier models by performing early fusion.
- Fusion doesn't need to be concatenation, but does need to happen early.

---
### References
- [The Visual QA Devil in the Details: The Impact of Early Fusion and Batch Norm on CLEVR](https://arxiv.org/pdf/1809.04482.pdf)
