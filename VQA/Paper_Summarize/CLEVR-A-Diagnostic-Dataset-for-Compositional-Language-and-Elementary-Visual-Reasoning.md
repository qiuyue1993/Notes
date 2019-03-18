## CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning

### Indexing:
- [Introduction](#Introduction)
- [The CLEVR Diagnostic Dataset](#The-CLEVR-Diagnositc-Dataset)
- [References](#References)
---
### Introduction
#### CLEVR Dataset
- We ensure that the information in each image is **complete and exclusive** so that external information sources, 
such as **commonsense knowledge, cannot increase the chance** of correctly answering questions. 
- We **minimize question-conditional bias** via rejection sampling within families of related questions, and **avoid degenerate questions** that are seemingly complex but contain simple shortcuts to the correct answer. 
-  we use **structured ground-truth representations** for both images and questions: images are annotated with ground-truth object positions and attributes, and questions are represented as functional programs that can be executed to answer the question

#### Weaknesses of VQA models
- **Short term memory**, such as comparing the attributes of objects.
- **Compositional reasoning**, such as recognizing novel attribute combinations.

---
### The CLEVR Diagnostic Dataset
- Images: synthetic images, associated with ground-truth object locations and attributes.
- Questions: automatically generated questions, associated with machine-readable form.
- Available for analyzing: question type, question topology (chain vs. tree), question length, relationship forms between objects.

#### Objects and relationships
Attributes:
- Shapes: cube, sphere, cylinder
- Sizes: small, large
- Materials: metal, rubber
- Colors: gray, red, blue, green, brown, purple, cyan, yellow

Spatial Relationships:
- left, right, behind, in front

Same Relationships:
- same

Attributes and Relationships also depend on relative object positions and camera viewpoint and context.

#### Scene representation
Scenes representation:
- Objects annotated with shape, size, color, material
- Object position on the ground-plane

#### Image generation
Image generation process:
- Randomly sampling a scene graph
- Rendering scene graph using Blender.

**Generation settings**:
- Every scene contains between three and ten objects with random attributes.
- No objects intersect
- Objects are at least partially visible.
- There are small horizontal and vertical margins between image-plane centers of each pair of objects.
- Positions of lights and camera are randomly jittered.

#### Question representation
- Each question in CLEVR is associated with a **functional program** that can be executed on an image's scene graph, yielding the answer to the question.
- Functional programs are built from simple basic functions that correspond to **elementary operations** of visual reasoning such as querying object attributes, counting sets of objects, or comparing values.
- We categorize questions by question type, **deﬁned by the outermost function** in the question’s program

#### Question families
Aim of Question families :
- Create questions in a finite number.
- Covert functional programs to natural language in a way that minimizes question-conditional bias.

Question families:
- A template for constructing functional programs
- Several text templates for expressing programs in natural language.
- 90 question families in total, each with a single program template and an average of four text templates.
- Use synonyms for each shape, color, and material.

#### Question generation
Question generation process:
- Choose a question family
- Select values for each of its template parameters
- Execute the resulting program on the image's scene graph to find the answer
- Use one of the text templates from the question family to generate the final natural-language question

Difficulties of question generation:
- Ill-posed questions
- Degenerate questions
- Bias

Solution:
- Depth-first search to find valid values for instantiating question families.
- Use ground-truth scene information to prune large swaths of search space.
- Rejection sampling to produce an approximately uniform answer distribution.

---
### References
- CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning[https://arxiv.org/pdf/1612.06890.pdf]
- SHAPES(https://arxiv.org/pdf/1511.02799.pdf)
