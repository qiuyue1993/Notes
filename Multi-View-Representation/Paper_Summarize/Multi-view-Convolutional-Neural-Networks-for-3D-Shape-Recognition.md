## Multi-view Convolutional Neural Networks for 3D Shape Recognition

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Method](#Method)
- [Experiments](#Experiments)
- [Conclusion](#Conclusion)
- [References](#References)

---
### Introduction
#### Abstract
- We ﬁrst present a standard CNN architecture trained to recognize the shapes’ rendered views independently of each other, and show that
a 3D shape can be recognized even from **a single view** at an accuracy far **higher than** using state-of-the-art **3D shape descriptors**.
- In addition, we present a novel **CNN architecture that combines information from multiple views** of a 3D shape into a **single and compact shape descriptor** offering even better recognition performance. 

#### Challenges of 3D representation
- Resolution
- Time consumption

#### Merits of 2D representation
- Leveraging the advances in image descriptors
- Massive image databases to pre-train CNN architectures

#### Multi-view CNN
- Multi-view CNN is related to **jittering** where transformed copies of the data are added during training to learn invariances to transformations such as rotation or translation.
- The multi-view CNN learns to **combine the views instead of averaging**, and thus can use the more informative views of the object for prediction while ignoring others. 

---
### Method



#### Input: A Multi-view Representation



#### Recognition with Multi-view Representations


#### Multi-view CNN: Learning to Aggregate Views


---
### Experiments
#### 3D Shape Classification and Retrieval


#### Sketch Recognition: Jittering Revisited


#### Sketch-based 3D Shape Retrieval



---
### Conclusion
- By building descriptors that are **aggregations of information from multiple views**, we can achieve **compactness, efﬁciency, and better accuracy**.
- In addition, by relating the content of 3D shapes to 2D representations like sketches, we can **retrieve these 3D shapes at high accuracy using sketches**, and leverage the implicit knowledge of 3Dshapes contained in their 2D views. 

---
### References
- [Multi-view CNN Paper](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf)

---
