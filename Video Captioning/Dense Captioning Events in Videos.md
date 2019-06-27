## Dense Captioning Events in Videos

### Indexing:
- [Introduction](#Introducton)
- [Dense-captioning events model](#Dense-captioning-events-model)
- [ActivityNet Captions dataset](#ActivityNet-Captions-dataset)
- [Experiments](#Experiments)
- [Thoughts](#Thoughts)
- [References](#References)
---
### Introduction
*Dense-Captioning Events*
- A new task of detecting and describing events in a video


*Illustration*

<img src="https://github.com/qiuyue1993/Notes/blob/master/Video%20Captioning/Images/Paper-Summarize_Dense-Captioning-Events_Task-Illustration.png" width="600" hegiht="400" align=center/>

---
### Dense-captioning events model
#### Overall Framework

<img src="https://github.com/qiuyue1993/Notes/blob/master/Video%20Captioning/Images/Paper-Summarize_Dense-Captioning-Events_Overall-Framework.png" width="600" hegiht="400" align=center/>

#### Challenges
- Detect **multiple events** in **short and long** video sequences
- Utilize the context from **past, concurrent and future** events to generate descriptions of each one

#### Intuition
- Introduces a variant of an existing proposal module that is designed to capture **both short as well as long events** than span minutes. Sample video frames **at different strides** and gather evidence to **propose events** at **different time scale**
- Introduces a new captioning module that **uses contextual information from past and future events** to jointly describe all events.

#### Definition
- Input: a sequence of video frame
- Ouptut: a set of sentences consists of the **start and end times**

#### Event proposal module
- Design an event proposal module to be **a variant of DAPs** that can detect longer events
- Firstly, a sequence of features are extracted from video
- Next, we sample the videos features at different strides and feed them into a proposal LSTM unit

#### Captioning module with context
- Design a captioning module to incorporate the "context" from its neighboring events
- Categorize all events into two buckets: past and future
- Concurrent events are split into one of the buckets according to the ending time
- Concatenate of $(h_{past}, h_i, h_{future})$ is fed into captioning LSTM

---
### ActivityNet Captions dataset
#### Statistics
- 20k videos
- Average 3.65 sentences with average length of 13.48 words per sentence
- Each sentence describes 36 seconds and 31% of video on average
- Entire paragraph describes 94.6% of video on average
- 10% Overlap of events
- 849 video hours
- 100k total descriptions, each with start and end time
- Videos as long as 10 minutes
- Centered around human activities vs Visual Genome (object-centric)

#### Challenges
- events can occur **within a second or last up to minutes**
- events are related to one another

---
### Experiments
#### Tasks
- Dense captioning events
- Localization
- Retrieval

#### Dense-captioning events
*Evaluation metrics*
- BLEU
- METEOR
- CIDEr

*With learnt proposals*
- B@1: 17.95
- B@2: 7.69
- B@3: 3.86
- B@4: 2.20
- METEOR: 4.82
- CIDEr: 17.29

---
### Thoughts
- Using graph to deal with the events relationships
- From video to dynamic graph
- Visual Discourse 
- Annotations of discourse: like visual genome
- **Diversity Captions Generation: 1video vs. 1 captions No problem??!! No biases??**

---
### References
- [Dense Captioning Events in Videos](https://arxiv.org/pdf/1705.00754.pdf)
- [Project](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
---
