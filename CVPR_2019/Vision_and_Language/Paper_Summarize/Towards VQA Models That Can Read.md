## Towards VQA Models That Can Read

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [LoRRA Look Read Reason Answer](#LoRRA-Look-Read-Reason-Answer)
- [TextVQA](#TextVQA)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
- [References](#References)

---
### Introduction
**TextVQA** dataset
- 45,336 questions
- 28,408 images from the Open Images dataset
- from categories tend to contain text: "billboard", "traffic sign", "whiteboard"

**Contributions**
- Introduce of a novel dataset **TextVQA**
- Introduce a novel model **LoRRA** that can reads text in the image and predicts the answer;
- Introduce OCR (Optical Character Recognition) into VQA 
- LoRRA outperforms existing methods on TextVQA as well as VQA 2.0

---
### Related Work
#### Visual Question Answering
- VQA Datasets: VQA 1.0, 2.0; DAQUAR; COCO-QA
- non-photo-realistic VQA datasets: CLEVR, NLVR, FigureQA
- Fact-Based VQA requires external knowledge

#### Text based VQA
- Datasets: COCO-Text; Street-View test IIIT-5k; ICDAR 2015 (do not involve answering questions)
- DVQA: VQA for graphs and plots
- Textbook QA (TQA): answering questions from middle-school textbooks
- MemexQA: VQA involves reasoning about the time and date

#### Visual Representations for VQA Models
- attention

#### Copy Mechanism
- The proposed model needs to decide whether the answer should be an OCR token detected
- Copy mechanism provides networks the ability to generate **out-of-vocabulary** words

---
### LoRRA Look Read Reason Answer
#### Overview
- Three components: VQA component; Reading component; Answering module
- The OCR model and backbone VQA model is arbitrary

#### VQA Component
- Question: embed the question by embedding function and RNN
- Images: spatial features
- Attention over spatial features
- Combine the output with question embedding

#### Reading Component
- OCR model: not jointly trained with VQA
- OCR models extracts words from the image and then embedded with pre-trained word embedding
- Use the same architecture as VQA component to get combined OCR-question features
- The architecture above do not share weights with VQA model

#### Answer Module
- Extend answer space to $N+M$ through addition of a dynamic component which corresponds to $M$ OCR tokens
- If the model predicts an index larger than $N$, copy the corresponding OCR token

#### Implementation Details
- VQA component: Pythia v1.0 (VQA 2018 challenge winner)

---
### TextVQA
#### Images
- Image source: Open Images v3 dataset 
- Three Stage pipeline for crowd-sourcing data

*First stage*
- annotators were asked to identify images that did not contain text
- Total images: 28,408

#### Questions and Answers
*Second stage*
- collect 1-2 questions for each image; 
- two questions would have different answers. 
- collect 10 answers for each question.

#### Statistics and Analysis
*Questions*
- Total questions: 45,336 questions (with 37,912 unique questions)
- Average question length: 7.18 (> VQA2.0(6.29), > VizWiz(6.68))
- Minimum question length: 3 words
- Questions often start with "what"
- Frequently inquiring about "time", "names", "brands" or "authors"

*Answers*
- 26,263 unique majority answers (49.2%) vs VQA 2.0 (3.4%) VS VizWiz(22.8%)
- Answer space is diverse, ("yes":4.71%)
- Average answer length: 1.58

---
### Experiments
#### Upper Bounds and Heuristics
*Vocabulary*
- **SA**: size of 3,996, contains answers which appear at least twice in training set
- **LA**: size of 8,000, containing the most frequent answers

*Evaluate different upper bounds of OCR module and benckmark biases in the dataset**
- **OCR UB**: Upper bound accuracy if the answer can be build directly from OCR tokens
- **LA UB**: the correct answer is present in **LA**
- **LA + OCR UB**: the correct answer is present either LA or OCR tokens
- **Rand 100**: selecting a random answer from top 100 most frequent answers
- **Wt. Rand 100**: the accuracy of baseline (iv) but with weightd random sampling using 100 most ocurring tokens' frequencies as weights
- **Majority Ans**: the accuracy of always predicting the "yes"
- **Random OCR token**: the accuracy of predicting a random OCR token from the OCR tokens detected in an image
- **OCR Max**: accuracy of always predicting the OCR token that is detected maximum times in the image

#### Baselines
- Question Only (Q)
- Image Only (I)

#### Ablations
- I+Q: State-of-the-art for VQA 2.0, doesn't use any kind of OCR features
- Pythia + O: Pythia with OCR but no copy module or dynamic answer space
- Pythia + O + C: Pythia with OCR but no fixed answer space

#### Experimental Setup
- Pytorch
- AdaMax optimizer for backpropagation
- Binary cross-entropy loss
- 24000 iterations with batch size of 128 on 8 GPUs
- Maximum question length: 14
- Maximum number of OCR tokens: 50
- All validation accuracies are averaged over 5 runs with different seeds

#### Results
- Human accuracy (85.01%) is consistent with VQA 2.0 and VizWiz
- OCR UB: 37.12%
- Random baselines, even the weighted one, are rarely correct
- The dataset does not have significant biases w.r.t. images and questions
- Inability of current VQA models to read and reason about text in images
- LoRRA(LA) with Pythia model outperforms all the ablations
- LoRRA can help state-of-the-art VQA models to perform better on TextVQA

---
### Conclusion
- **TextVQA** dataset
- Model: **LoRRA**, agnostic to the specific of the underlying **OCR and VQA** models

---
### References
- [Towards VQA Models That Can Read](https://arxiv.org/pdf/1904.08920.pdf)
- [Code](https://github.com/facebookresearch/pythia)
- [Program Page](https://textvqa.org/)
---
