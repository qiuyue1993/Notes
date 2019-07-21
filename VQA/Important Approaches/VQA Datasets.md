# VQA Datasets

## Indexing:
- [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](#GQA)
- [References](References)
---
## GQA
- Accept to CVPR 2019

### Introduction

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Illustration.png" width="600" hegiht="400" align=center/>

*Intuition*
- For **real-world visual reasoning**
- And **compositional** question answering

- Dataset example

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Dataset-Examples.png" width="600" hegiht="400" align=center/>

*GQA*
- Images come from Visual Genome dataset
- 1.7M questions after balancing
- Every question is generated from scene graph
- Every question have a function program that present questions' semantics
- Answer distribution and question biases are well controlled
- A series of evaluation metrics are proposed in this paper

*Contributions*
- The GQA dataset as a resource for **studying visual reasoning**
- Development of an effective method for generating a large number of semantically varied question, which marries scene graph representations with computational linguistic methods
- New **metrics for VQA** that allow for better assessment of system success and failure modes


### The GQA Dataset

- Dataset Generation Processes

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Dataset-Generation-Processes.png" width="800" hegiht="400" align=center/>


#### Overview
*GQA can be used to measure performance on*
- Object and attribute recognition
- Transitive relation tracking
- Spatial reasoning
- Logical inference and comparisons
- and so on

*Four-step dataset construction pipeline*
- Thoroughly clean, normalize, consolidate and augment the Visual Genome scene graphs
- Travese the objects and relations within the graphs and marry them with **grammatical patterns and sundry probabilistic grammer rules** to produce a semantically-rich and diverse set of questions
- Use the underlying semantic forms to **reduce biases**
- Discuss the question functional representation, and explain how we use it to **compute entailment between questions**, supporting new **evaluation metrics**

#### Scene Graph Normalization
*Scene Graph annotations of Visual Genome Dataset*
- Scene Graph: A formalized representation of image
- Nodes: each node denotes an object, linked to a bbox 
- Attributes: each object have about 1-3 attributes (e.g., color, shape, material, activity)
- Relation edges: represents actions (verbs), spatial relations (prepositions), and comparatives
- 113k images from COCO and Flickr

*Clean steps of Scene Graphs for GQA*
- Create a clean, consolidate and unambiguous ontology over the graph with **2690 classes** including objects, attributes and relations
- Prune inaccurate or unnatural edges
- Enrich the graph with positional information (absolute and relative) and semantic properties (location, weather)

#### The Question Engine

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Programs.png" width="600" hegiht="400" align=center/>


*Two resources*
- Scene graphs (rich content of objects, attributes and relationships)
- Structural patterns

*Statistics*
- 524 patterns
- 117 question groups
- 1878 answers 

*Question Group*
- Each group is associated with 3 components:
- A **function program** that represents its semantics
- A set of textual rephrase: e.g., "What (Which) $\langle$ type $\rangle$ [do you think] $\langle$ is $\rangle$ $\langle$ theObject $\rangle$ ?"
- A pair of short and long answers: e.g., $\langle$ attribute $\rangle$ and "The $\langle$ object $\rangle$ $\langle$ is $\rangle$ $\langle$ attribute $\rangle$"
 
*Question Diversity*
- Introduce synonyms
- Incorporate probabilistic sections into the patterns

*Noteworthy*
- $\langle$ theObject $\rangle$ can be long sentences
  
*Four resources of generating questions*
- The clean scene graphs
- The structural patterns
- The object references
- The decoys

*Question Generation*
- Selecting and instantiating question pattern, such as: "What (color) (is) the (apple on the table), (red) or (green)"
- By the end of current stage, they obtained a question set with **22M questions**

#### Functional Representation and Entailment
*Function Program*
- Each question pattern is associated with a structured representation 
- E.g., For question "What color is the apple on the white table?"
- The program is: "table->filter:white->relate(subject,on): apple->query: color"
- Each program can be solved by an atomic operations

#### Sampling and Balancing
*Main issues of existing VQA datasets*
- Question-conditional biases

*GQA's effort to avoid biases*
- Using the functional programs attached to each question to smooth out the answer distribution

*Process*
- Devide two labels, global and local
- Global label: assigns the question to its answer type, e.g. *color* for *What color is the apple?*
- Local label: considers the main subject/s of the question, e.g. *apple-color* or *table-material*
- Smoothing answer distribution on above two degrees

### Analysis and Baseline Experiments
#### Dataset Analysis and Comparison

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Dataset-Statistics.png" width="600" hegiht="400" align=center/>


*Statistics*
- 22,669,678 questions
- 113,018 images
- vocabulary size: 3.097 words
- 1,878 possible answers 

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Detailed-Question-Type-Statistics.png" width="400" hegiht="150" align=center/>

*Two types*
- Structural type: derived from the final functional program of the questions
- Structural types: **verify** for yes/no questions; **query** for all open questions; **choose** for questions that represent two alternatives; **logical** which involve logical inference; **compare** for comparison questions between two or more objects; 
- Semantic type: refers to the main subject of the question
- Semantic types: **object** for existence questions; **attribute** consider the properties or position of an object; **category** related to object identification within some class; **relation** for questions asking about the subject or object of a described relation; **global** about overall properties of the scene such as weather or place.

#### Baseline Experiments

<img src="https://github.com/qiuyue1993/Notes/blob/master/VQA/images/Paper-Summarization_GQA_Benchmark-results.png" width="600" hegiht="400" align=center/>

*Baseline methods*
- "Blind" LSTM
- "Deaf" CNN
- LSTM + CNN
- Prior model 1 based on local question group
- Prior model 2 based on global question group
- Up-down
- MAC

*Results*
- Human exceed all methods above by a large margin


#### Transdfer Performance
- Test the transder performance between the GQA and VQA datasets
- The results show that the GQA can serve as a good proxy for human-like questions

#### New Evaluation Metrics
- Introduce 5 new metrics

***Consistency***
- This metric measures **consistency across different questions**
- Which is evaluated  by measuring the model's accuracy over the questions and their entailed questions


***Validity*** and ***Plausibility***
- Validity: checks whether a given answr is in the question scope, e.g., responding color to a color question
- Plausibility: measuring whether the answer is reasonable, or makes sense (e.g. elephant usually do not eat, say, pizza); (Evaluated by checking whether the answer occurs at least once in relation with the question's subject, across the whole dataset)

***Distribution***
- Measures the overall match between the true answer distribution and the model predicted distribution, using Chi-Square statistic

***Grounding***
- For attention-based models, the grounding score checks whether the model attends to regions within the image that are relevant to the question



### Comments
- Scene Graphs in GQA have object with 0 attributes?

---
## References
- [GQA Project Site](https://cs.stanford.edu/people/dorarad/gqa/)
- [GQA Paper](https://arxiv.org/pdf/1902.09506.pdf)
---
