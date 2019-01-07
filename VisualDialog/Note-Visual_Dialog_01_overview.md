## The Visual Dialog Note - 01: Overview of Visual Dialog

### Indexing: 
- [Introduction](#Introduction)
- [Datasets](#Datasets)
- [Evaluation](#Evaluation)
- [Challenges](#Challenges)
- [Encoder Decoder Models](#Encoder-Decoder-Models)
- [Performance](#Performance)
- [Summary](#Summary)
- [Reference](#Reference)
---

###  Introduction:
<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/visualdialog_01-overview.png" width="200" hegiht="200" align=center />
#### Why Visual Dialog?
- Natural Language instructions for robots
- Aid visually impaired users
- Aid  ``situationally impaired'' users

#### Visual Dialog: Task
##### Given:
- Image I
- Human dialog history $(Q_1, A_1), (Q_2, A_2), ..., (Q_{t-1}, A_{t-1})$
- Follow-up question $Q_t$
##### Predict:
- free-form natural language answer

---
### Datasets:
#### VisDial v1.0
- ~130k images from   COCO
- 1 dialog/image
- 10 rounds of question-answers/dialog
- Total ~1300k dialog question-answers

---
### Evaluation
#### Evaluation(as in CVPR17 paper)
- 100 answer options
- 50 answers from NN questions
- 30 popular answers
- up to 20 random answers
- Rank 100 options
- Accurarcy: mean rank of GT answer, recall@k
#### Normalized Discounted Cumulative Gain (NDCG)
```
Answer options: [``two'', ``yes'', ``probably'', ``no'', ``yes it is'' ]
Ground-truth relevances: [0, 1.0, 0.5, 0, 1.0]

Ideal ranking of answer options: [``yes'', ``yes it is'', ``probably'', ``two'', ``no'']
Submitted ranking of answer options: [``yes'', ``yes it is'', ``two'', ``probably'', ``no'']
```
Then, 
$$DCG@k = \sum_{i=1}^{k}\frac{relevance_i}{\log_{2}{i+1}}$$
And,
$$NDCG@k =\frac{DCG@k\ for\ submitted\ ranking}{DCG@k\ for\ ideal\ ranking}$$

##### Features for NDCG:
- Swapping options with same relevance does not affect NDCG
- Shuffling options after first k ranks does not affect NDCG

---
### Challenges
#### EvalAI
- [Visual Dialog Challenge](https://visualdialog.org/challenge/2018)
- [WizWiz Challenge](http://vizwiz.org/data/#challenge)
- [Vision and Language Navigation](https://evalai.cloudcv.org/web/challenges/challenge-page/97/overview)
- [VQA Challenge](https://visualqa.org/challenge.html)
---
### Encoder Decoder Models
#### Decoders:
##### Generative Decoding
- During training, maximizes likelihood of GT human response
- During evaluation, ranks options by LL scores

##### Discriminative Decoding
- Computes dot product between input encoding and LSTM
- Encoding of each of 100 options
---
### Performance
![VD_Performance](https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/visualdialog_01-challengeresults.png)

- Human performance: NDCG 64.27
- 1st place model: 55.64
- baseline: 47.57
- Low performance on counting
- All teams perform better on binary
- Gap wider for generative than discriminative
---
### Summary
- 8% NDCG improvement above strong baselines
- 9% below human NDCG
- lowest performance at counting (``how many'') questions
- 7 out of 11 teams used object detection features 

---
### Reference
- [Visual Dialog](https://visualdialog.org/)
- [2018_09_08_ECCV_VisDial_Challenge.pdf - Google ドライブ](https://drive.google.com/file/d/1xSzg8mJYPNSRXtkCRpduefplrrUTe0PW/view)
