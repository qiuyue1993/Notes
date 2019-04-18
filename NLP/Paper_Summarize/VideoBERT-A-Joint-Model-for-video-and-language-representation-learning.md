## VideoBERT: A Joint Model for Video and Language Representation Learning

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [VideoBERT](#VideoBERT)
- [Experiments and Analysis](#Experiments-and-Analysis)
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
#### The BERT model
- The BERT model can be thought of as a **fully connected Markov Random Field (MRF)** on a set of discrete tokens.
- [SEP] token indicates the end of the sequence.

#### The VideoBERT model
- In oder to leverage pretrained language models and scalable implementations for inference and learning, VideoBERT **transform the raw visual data into a discrete sequence of tokens**.
- Combine the linguistic sentence with the visual sentence to generate data such as: [CLS] orange chicken with [MASK] sauce [>] v01 [MASK] v08 v72 [SEP]. ([>] is a special token to combine text and video sentences)
- VideoBERT propose a linguistic-visual alignment task, where they use the final hidden state of the [CLS] token to predict whether the linguistic sentence is **temporally aligned** with the visual sentence.

**steps for learning temporally alignment**
- Randomly concatenate neighboring sentences into a single long sentence, to allow the model to learn **semantic correspondence.
- Randomly pick a subsampling rate of 1 to 5 steps for the video tokens (merits: robust to variations in video sppeds; allows the model to capture temporal dynamics over greater time horizons and learn longer-term state transitions)

**training regimes**
- text-only, video-only: the standard mask-completion objectives are used for training the models.
- text-video: use the linguistic-visual alignment classification objective.
- Overall training objective: weighted sum of the individual objectives.
- text objective: forces VideoBERT to learning language modeling
- video objective: forces VideoBERT to learn a language model for video, which can be used for learning dynamics and forecasting
- text-video objective: forces VideoBERT to learn a correspondence between the two domains.

**Applications**
- Treat VideoBERT as a probabilistic model: ask it to predict or impute the symbols that have been MASKed out.
- Extract the predicted representation for the [CLS] token, and use that dense vector as a representation of the entire input.

---
### Experiments and Analysis
#### Dataset
- Find videos where the spoken words are more likely to refer to visual content.
- Turn to YouTube to collect a large-scale video dataset for training (cooking and recipe).
- 312K videos, 23,186 hours total duration
- ASR (Automatic speech recognition) toolkit to retrieve timestamped speech information
- Result of 120K english dataset
- Evaluate VideoBERT on the YouCook 2 dataset.

#### Video and Language Preprocessing
- For each video, sample frames at 20 fps, create clips from 30-frame non-overlapping windows over the video
- For each 30-frame clip, apply a pretrained video ConvNet to extract features (S3D).
- Take the feature activations before the final linear classifier and apply 3D average pooling to obtain a 1024-dimension feature vector.
- Tokenize the visual features using hierarchical k-means, 20736 clusters in total.
- For each ASR word sequence, break the stream of words into sentences by adding punctuation using an off-the-shelf LSTM-based language model.

#### Model Pre-training


#### Zero-shot action classification


#### Benefits of large training sets


#### Transfer learning for captioning



---
### Conclusion
- The paper adapts the powerful **BERT** model to learn a joint **visual-linguistic representation** for video.
- Experimental results demonstrate that VideoBERT can learn **high-level semantic representations**
- The model can be used for **open-vocabulary classification** and its performance grows monotonically with the **size of training set** 

---
### References
- [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/pdf/1904.01766.pdf)
- [Summarize Blog 1](https://zhuanlan.zhihu.com/p/62642374)
