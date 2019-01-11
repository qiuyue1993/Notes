## Paper Summarize - Visual Dialog

### Indexing
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [VisDial Dataset](#VisDial-Dataset)
- [VisDial Dataset Analysis](#VisDial-Dataset-Analysis)
- [Neural Visual Dialog Models](#Neural-Visual-Dialog-Models)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
- [Reference](#Reference)
---
### Introduction
#### Task
- Given an image, a dialog history, and a question about the image, the agent has to ground the question in image, infer context from history, and answer the question accurately.

#### Models
- We introduce a family of neural encoder-decoder models for Visual Dialog with 3 encoders – Late Fusion, Hierarchical Recurrent Encoder and Memory Network – and 2 decoders (generative and discriminative), which outperform a number of sophisticated baselines. 

#### Evaluation
- We propose a retrievalbased evaluation protocol for Visual Dialog where the AI agent is asked to sort a set of candidate answers and evaluated on metrics such as mean-reciprocal-rank of human response.

#### Applications
- Aiding visually impaired users in understanding their surroundings or social media content
- Aiding analysts in making decisions based on large quantities of surveillance data
- Interacting with an AI assistant
- Robotics applications (e.g. search and rescue missions) where the operator may be ‘situationally blind’ and operating via language

#### Former language-only tasks
- Goal-driven dialog: The shorter, the better
- Goal-free dialog: The longer, the better
---

### Related Work
#### Vision and Language:
- Image Captioning
- Video/movie description
- Text-to-image coreference/grounding
- Visual storytelling
- Visual Question Answering (VQA)

#### Visual Turing Test:
-  A system that asks templated, binary questions. 

#### Text-based Question Answering:
- VisDial can be viewed as a fusion of reading comprehension and VQA.
- 30M Factoid Question-Answer corpus
- 100K SimpleQuestions dataset
- DeepMind Q&A dataset
-  bAbI dataset
-  SQuAD dataset for reading comprehension

#### Conversational Modeling and Chatbots:
- A recent large-scale conversation dataset: Ubuntu Dialogue Corpus
- **One important difference between free-form textual dialog and VisDial is that in VisDial, the two participants are not symmetric**

---
### VisDial Dataset
Collecting visual dialog data on COCO images.

#### Live Chat Interface
- Paired workers on AMT, real-time chat.
- Questioner:  Ask questions about image from the image caption given. (Image remain hidden)
- Answerer: Answer questions. (Both image and caption are been shown)
- Unconstrained 'live' chat, as naturally and 'conversationally' as possible.

#### Building a 2-person chat on AMT
- AMT is simply not designed for multi-user Human Intelligence Tasks.

---
### VisDial Dataset Analysis

#### VisDial v0.9
- 123k images from COCO.
- 1 dialog (10 QA pairs) on each image.

#### Analyzing VisDial Questions
- Without visual priming bias.
- Most questions range from 4 to 10 words.
- Most frequent question : "Is ... "
- More binary questions than VQA.
- Comparing with VQA datasets, questions are asking to build a mental model of the scene

#### Analyzing VisDial Answers
- Longer than VQA datasets, with mean-length 2.9 words.
- Long tail distribtution.
- Some answers express doubt, uncertainty or lack of information. However, in VQA datasets, they don't have this type of answer.
- For binary questions, usually answers are not just yes or no, always contains additional information or clarification. 

#### Analyzing VisDial Dialog
##### Coreference in dialog.
- 38% questions, 19% answers, 98% dialogs contain at least one pronoun.

##### Temporal Continuity in Dialog Topics.
- 4.55 topics on average. (Across 10 rounds)

#### VisDial Evaluation Protocol
Evaluate individual responses at each round in a retrieval or multiple-choice setup.
- Rank of human response
- Recall @k (Existence of human response in top-k ranked responses)
- Mean Reciprocal Rank (MRR) of human response   

Four kinds of candidate answers
- Correct
- Plausible (answers to most similar questions)
- Popular
- Random
 
---

### Neural Visual Dialog Models
#### Problem Definition
Input:
-  Image *I*
- ``ground-truth'' dialog history $H = (C, (Q_1, A_1), ... , (Q_{t-1}, A_{t-1}))$
- Question $Q_t$
- Candidate answers $A_t =\{A_{t}^{(1)}, ... , A_{t}^{(100)}\}$

Output:
- Sort of $A_t$

Base framework:
- Encoder-Decoder framework
- Encoder: convert input $(I, H, Q_t)$ into a vector space.
- Decoder: convert the embedded vector into output.

#### Decoders
##### Generative (LSTM) decoder
- Initial state of LSTM: encoded vector
- Training: maximize the log-likelihood of gt answer squences
- Evaluating: using log-likelihood scores to rank candidate answers.
- **This kind of models does not exploit the biases in potion creation.** 

##### Discriminative (softmax) decoder
- Computes dot prodcution similarity between input encoding and an LSTM encoding of each answer.
- Training: maximize the log-likelihood of the correct option.
- Evaluation: options are ranked based on their posterior probabilites.

#### Encoders
Image Representation: $l2$-normalized activations of the penultimate layer of VGG-16.
For each decoder, experiment with all possible ablated versions: $E(Q_t), E(Q_t, I), E(Q_t, H), E(Q_t, I, H)$
##### Late Fusion (LF) Encoder:
- LSTM for $H$: treat $H$ as a long string $(H_0, ... , H_{t-1})$
- LSTM for $Q_t$
- Concatenate the representation of $I, H, Q_t$ 
- Linearyly transformed to joint embedding size.

##### Hierarchical Recurrent Encoder (HRE):
- Recurrent block ($R_t$): Embeds question and image jointly via LSTM.
- Recurrent block ($R_t$): Embeds each rounds of history $H_t$
- Concatenate above embeddings
- Pass the embeddings to Dialog-RNN
- Attention-over-history: a softmax over previous rounds, to choose the history relevant to the current question.
##### Memory Network (MN) Encoder:
- LSTM Encoder for $Q_t$: obtain 512-d vector
- LSTM Encoder for $(H_0, ... , H_{t-1})$: obtain $t \times 512$ matrix
- Inner production of representation of $Q_t$ and $(H_0, ... , H_{t-1})$
- Fed above result to softmax to get attention-over-history probabilities.
- Weighted sum of representation of  $(H_0, ... , H_{t-1})$ and fed into a fc layer
- Add above result with the representation of  $Q_t$ 

---

### Experiments
#### Splits
- Training: 80k dialogs
- Validation: 3k
- Testing: 40k

#### Baselines
- Answer Prior: linear classifier on Answer options
- NN-Q: find $k$ nearest neighbor questions and score answer options by their mean-similarity with these $k$ answers
- NN-QI: filtering the $k$ nearest neighbor questions by the image similarity
- VQA models such as, SAN, HieCoAtt

#### Results
- All learning based models outperform non-learning baselines.
- All discriminative models outperform generative models
- MN-QIH-G : 0.526 MRR
- MN-QIH-D: 0.597 MRR
- Models naively incorporating history doesn't help much.
- Models better encode history (MN/HRE) perform better than LF models with/without history.
- Models looking at $I$ outperform corresponding blind models.

---

### Conclusion
- A new AI task - Visual Dialog
- Large-scale dataset - VisDial
- Retrieval-based evaluation protocol
- A family of encoder-decoder models
- Significant scope between human and proposed methods for future development 
---
### Reference
- [Visual Dialog](https://arxiv.org/pdf/1611.08669.pdf)
