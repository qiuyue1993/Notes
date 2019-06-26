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



---
### Dense-captioning events model
#### Intuition
- Introduces a variant of an existing proposal module that is designed to capture **both short as well as long events** than span minutes.
- Introduces a new captioning module that **uses contextual information from past and future events** to jointly describe all events.

#### Abstract
- Sample video frames **at different strides** and gather evidence to **propose events** at **different time scale**

---
### ActivityNet Captions dataset
#### Statistics
- 20k videos
- 849 video hours
- 100k total descriptions, each with start and end time
- Videos as long as 10 minutes, annoatated with on average 3.65 sentences.
- Centered around human activities

#### Challenges
- events can occur **within a second or last up to minutes**
- events are related to one another

---
### Experiments


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
