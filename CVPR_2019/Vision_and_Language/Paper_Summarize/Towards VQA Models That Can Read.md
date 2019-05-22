## Towards VQA Models That Can Read

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [LoRRA Look Read Reason Answer](#LoRRA-Look-Read-Reason-Answer)
- [TextVQA](#TextVQA)
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


#### Questions and Answers



#### Statistics and Analysis



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
