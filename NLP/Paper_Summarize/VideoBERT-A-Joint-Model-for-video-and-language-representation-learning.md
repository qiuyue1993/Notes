## VideoBERT: A Joint Model for Video and Language Representation Learning

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [VideoBERT](#VideoBERT)
- [Conclusion](#Conclusion)
- [References](#References)

---
### Introduction
- Unsupervised Learning using Youtube data
- ASR: Generating text from video
- Video representation: S3D, hierachy clustering

**Abstract**
- We propose a **joint visual-linguistic model** to learn high-level features **without any explicit supervision**.
- We build upon the BERT model to learn **bidirectional joint distributions** over sequences of visual and linguistic tokens, derived from vector quantization of video data and off-the-shelf speech recognition outputs, respectively.

**VideoBERT**

Model the relationship between the visual domain and the linguistic domain by combining three off-the-shelf methods:
- ASR: automatic speech recognition system to convert speech to text
- Vector Quantization (VQ): applied to low-level spatio-temporal visual features derived from pretrained video classification models.
- BERT: learning joint distributions over sequences of discrete tokens.

**Applications**
- Text-to-video prediction
- Video-to-text task of dense video captioning
- Longrange forecasting (the model can be used to generate plausible guesses at a much **higher level of abstraction** compared to other deep generative models for video)

**Contribution**
- A simple way to learn **high level video representations** that capture **semantically** meaningful and **temporally long-range structure**.

**Tips**
- VideoBERT can be applied directly to **open vocabulary classification**.
- **Large amounts of training data** and cross-modal information are critical to performance.

---
### Related-Work
**Supervised learning**
- Supervised learning of Video representation have limits in time (usually a few seconds long)
- Cost of collect labeled data

**Unsupervised learning**
- VideoBERT use a Markov Random Field model instead of stochastic latent variables.

**Self-supervised learning**
- Self-supervised learning partition the signal into two or more blocks, such as gray scale and color, or previous frame and next frame, and try to predict one from the other.
- VideoBERT is similar, except they quantized visual words instead of pixels.

**Cross-modal learning**
- Videos contain synchronized audio and visual signals, the two modalities can supervise each other to learn strong self-supervised video representation.
- In VideoBERT, they use speech rather than low-level sounds as a source of cross-modal supervision.

**Natural language models**
- VideoBERT extend the BERT model to capture structure in both the linguistic and visual domains.

**Image and video captioning**

**Instructional videos**
- VideoBERT does not use any mamual labeling and learn a large-scale generative model of both words and visual signals.

---
### VideoBERT


---
### Conclusion
- The paper adapts the powerful **BERT** model to learn a joint **visual-linguistic representation** for video.
- Experimental results demonstrate that VideoBERT can learn **high-level semantic representations**
- The model can be used for **open-vocabulary classification** and its performance grows monotonically with the **size of training set** 

---
### References
- [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/pdf/1904.01766.pdf)
- [Summarize Blog 1](https://zhuanlan.zhihu.com/p/62642374)
