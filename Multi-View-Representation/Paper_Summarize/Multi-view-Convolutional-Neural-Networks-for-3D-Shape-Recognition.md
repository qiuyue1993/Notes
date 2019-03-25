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
### Related Work
#### Shape descriptors
##### 3D shape descriptors 
- Directly work on native 3D representations of objects, such as polygon meshes, voxel-based discretizations, point clouds, or implicit surfaces.
**Former works:**
- Voxel-based representation through 3D convolutional nets
- Histograms of bag-of-features models constructed out of surface normals, curvatures and so on.
- Challenges of hand-designed 3D features: lack of datasets and size limitation of datasets; 3D descriptors tend to be very high-dimensional, making classifiers prone to overfitting.

##### Collection of 2D projections
- LightField descriptor, which extracts a set of geometric and Fourier descriptors from object silhouettes rendered from several different viewpoints.
- Former descriptors are largely **hand-engineered** and some do not generalize well across different domains.

---
### Method
- If the views are rendered **in a reproducible order**, one could also **concatenate the 2D descriptors of all** the views.
- Our approach is to learn to **combine information from multiple views** using a uniﬁed CNN architecture that includes a **view-pooling layer**.

#### Input: A Multi-view Representation
##### Camera Setup
**First camera setup**
- Assume that the input shapes are **upright oriented** along a consistent axis (e.g., z-axis)
- 12 rendered views by placing 12 virtual cameras around the model every 30 degrees.
- The cameras are elevated 30 degrees from the ground plane.
- The cameras are pointing towards the centroid of the model.

**Second camera setup**
- Do not make use of the assumption about consistent up-right orientation of shapes.
- 20 virtual cameras, 4 rendered views from each camera, yielding total 80 views.

#### Recognition with Multi-view Representations
**Image descriptors**
- Two types of image descriptors for each 2D view: image descriptor based on Fisher vectors with multiscale SIFT; CNN activation features.
- CNN features: Use the VGG-M network, consists of mainly five convolutional layers conv_{1,...,5} followed by three fully connected layers fc_{6,...,8} and a softmax classification layer. Use **fc_7** as image descriptor.

**Classification**
- **Sum up** the values over 12 views with the highest sum.
- **Averaging** image descriptors lead to **worse** accuracy.

**Retrieval**
- A distance or similarity measure is required for retrieval tasks.
- Using l_2 distance between image feature vectors as distance.

#### Multi-view CNN: Learning to Aggregate Views
- We design the multi-view CNN on top of image-based CNNs;
- CNN_1: Each image of a 3D shape's multi-view representation is passed through CNN_1 seperately and **shared weight**.
- View-pooling: Aggregate multi-view information. We use **element-wise maximum operation**. **Element-wise mean operation is not as effective.** **placed close to the last conv layer resulted in optimal performance.**
- CNN_2


**Low-rank Mahalanobis metric:**
- Used for better performance on retrieval task.

---
### Experiments
#### 3D Shape Classification and Retrieval
- ModelNet40 dataset
- Baselines: 3D ShapeNets; Spherical Harmonics descriptor; LightField descriptor; Fisher vectors
- Assuption of **upright orientation didn't affect** the results.
- Performance is **not very sensitive among the later few layers** (conv_4-fc_7)

#### Sketch Recognition: Jittering Revisited
- We now examine whether we can get more **beneﬁt out of jittered views of an image** by using the same feature aggregation scheme we developed for recognizing 3D shapes.

#### Sketch-based 3D Shape Retrieval

---
### Conclusion
- By building descriptors that are **aggregations of information from multiple views**, we can achieve **compactness, efﬁciency, and better accuracy**.
- In addition, by relating the content of 3D shapes to 2D representations like sketches, we can **retrieve these 3D shapes at high accuracy using sketches**, and leverage the implicit knowledge of 3Dshapes contained in their 2D views. 

---
### References
- [Multi-view CNN Paper](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf)

---
