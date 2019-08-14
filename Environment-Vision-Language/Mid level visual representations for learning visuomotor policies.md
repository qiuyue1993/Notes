# Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies

## Indexing:
- [Introduction](#Introduction)
- [Approach](#Approach)
- [Experiments](#Experiments)
- [Comments](#Comments)
- [References](#References)
---
## Introduction
**Abstract**
- Discuss how much does **having visual priors abouth the world** assist in learning to perform **downstream motor tasks**
- By **integrating a generic perceptual skill set** within a **reinforcement learning framework**

**Finding and Approach**
- Finding: The correct choice of feature depends on the downstream task.
- Approach: introducing a **principled method for selecting a general-purpose feature set**
- Results: higher final performance with at least an order of magnitude less data than learning from scratch

**Simplifying assumptions**
- Selected **Locomotive Tasks** as the active tasks, and discussed the utility of mid-level vision on it
- Limitations of existing RL methods: difficulties in **long-range exploration** and **credit assignment with sparse rewards**
- Relaxing the **fixed set of mid-level features constraint** may improve the performance
- **Lifelong learning** of updating the visual estimators are important future research questions.

**Problems of existing RL**
- Previous research on RL of **pixel-to-torque** raised a question that **if all one needs from images can be learned from scratch using raw pixels by RL?**
- Requires massive amounts of data, resulting policies exhibit difficulties reproducing across environments with even modest visual differences

**Proposals**
- Including appropriate perceptual priors can alleviate these two phenomena, improving generalization and sample 
- We study how much **standard mid-level vision tasks and their associated features** can be used with **RL frameworks** in order to **train effective visuomotor policies**
- Analysis three questions: learning speed; generalization to unseen test spaces; whether a fixed feature could suffice or a set of features is required for supporting arbitrary motor tasks.
- We put forth a simple and practicle solver that takes a large set of features and outputs a smaller feature subset that minimizes the worst-case distance between the selected subset and the best-possible choice

**Findings**
- Appropriate **choice of feature** depends fiercely on the final task. Solving multiple active tasks therefore requires a set of features.

---
## Approach
### Overall
- Our setup assumes access to a set of features, where each feature is a function that can be applied to raw sensory data

### Using Mid-Level Vision for Active Tasks
- Choose 20 mid-level visual task 
- Freeze each encoder's weights to transform each observed image
- During training, only the agent policy is updated

### Core Questions
- **Sample Efficiency**: examine if an agent can learn faster
- **Generalization**: can agent generalize better to unseen spaces
- **Single Feature or Feature Set**: can a single feature support all downstream tasks? Is a set of features requried for gaining the feature benefits on arbitrary active tasks?
- **Findings**: depth estimation features perform well for visual exploration and object classification for target-driven navigation

### A Covering Set for Mid-Level Perception
- A compact feature set is desirable; therefore, we propose a **Max-Coverage Feature Selector** that curates a compact subset of features
- **With a measure of distance between features, we can explicitly minimize the worst-case distance between the best feature and out selected subset**; The taxonomy method defines exactly such a distance.

---
## Experiments
### Overall
- 20 vision features, 4 baselines, 3-8 seeds per scenario
- 800 policies
- 109,639 GPU-hours

### Experimental Setup
*Environments*
- Gibson environment (perceptually similar to the real world)

*Train/Test Split*
- Train and test in two disjoint sets of buildings
- Training space: 40.2 square meters
- Testing space: 415.6 square meters
- For local planning and exploration, train/test cover 154.9 and 1270.1 square meters

#### Downstream Active Tasks
*Navigation to a Visual Target*
- Task: locate a specific target object as fast as possible
- Touching the target: one-time positive reward +10
- Otherwise, small penalty -0.025 for living
- Maximum episode length: 400
- Shortest path: avg. 30 steps

*Visual Exploration*
- Must visit as many new parts of the space as quickly as possible
- The environment is partitioned into small occupancy cells
- Reward: proportional to the number of newly revealed cells
- Maximum episode length: 1000

*Local Planning*
- The agent must direct itself to a given nonvisual target destination using visual inputs, avoid obstacles and walls
- Receives dense positive reward proportional to the progress it makes toward the goal, penalized for colliding with walls and objects
- A small negative reward for living
- Maximum episode length: 400

*Observation Space*
- RGB image
- Minimum amount of side information

*Action Space*
- Low-level controller for robot actuation
- Actions: turn left, turn right, move forward

#### Mid-Level Features
- 20 different computer vision tasks
- texture-based tasks (denoising)
- 3D pixel-level (depth estimation)
- low-dimensional geometry (room layout)
- semantic tasks (object classification)
- Use pre-trained network of taskonomy with ResNet-50 encoder without a global average-pooling layer
- All networks use identical hyperparameters

#### Reinforcement Learning Algorithm
- Use the common Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation
- Using experience replay and off-policy variant of PPO
- For each task and environment, conduct a hyperparameter search optimized for the *scratch* baseline and reuse it for every feature

### Baselines
*Tabula Rasa (Scratch) Learning*
- Trains the agent from scratch. Receives the raw RGB images as input and uses a randomly **initialized AtariNet tower**

*Blind Intelligent Actor*
- Similiar with *Scratch*, except that the visual input is a fixed image and does not depend on the state of the environment

*Random Nonlinear Projections*
- Identical to using *mid-level* features, except that the encoder network is randomly initialized and then frozen

*Pixels as Features*
- Identical to using *mid-level* features, except that they downsample the input image to the same size as the features and use it as the feature

*Random Actions*
- Uniformly randomly samples from the action space

*SOTA Feature Learning*
- Dynamic modeling
- Curiosity
- DARLA
- ImageNet pretraining

### Experimental results on hypothesis testing
#### Hypothesis 1: Sample Complexity Results
- For each of the active tasks, several feature-based agents learn significantly faster than *scratch*
- Policy trained with object classification recognizes and converges on the navigation target, but fails to cover the entire space in exploration
- Distance estimation features only help the agent cover nearly the entire space in exploration, but fail in navigation 

#### Hypothesis 2: Generalization Results
*Large-Scale Analysis*
- For all task, there is a significant gap between train/test performance for scratch, and a much smaller one for the best feature

*Mind the Gap*
- Agents trained from scratch seem to **overfit completely**, rarely doing better than blind agents in the test environment

#### Hypothesis 3: Rank Reversal Results
- We found that there may not be one or two single features that consistently outperform all others
- The top-performing exploration agent used *Distance Estimation features*
- The top navigation agent used *Ojbect Classification features*
- The trend of **rank reversal** appears to be a widespread phenomenon
- The **semantic features are useful for navigation**
- The **geometric features are useful for exploration**
- The SOTA representation learning methods are similarly task-specific, but the best feature outperforms them by a large margin

### Max-Coverage Feature Set Analysis
*Overview*
- Use feature set (with size $k=1 to 4$)

*Performance*
- With $k=4$ features, proposed max-coverage feature set can match or exceed the performance of the *best* task-specific feature

*Sample Efficiency*
- The proposed max-coverage feature set expected to be worse. However, they did not find a noticeable difference

*Practicality*
- The structure lends itself well to model-parallelism on modern computatinal architectures

### Universality Experiments
*Universality in Additional Buildings*
- Repeat testing in 9 other buildings 
- **Spearman's p of 0.93 for navigation and 0.85 for exploration**

*Universality in Additional Simulators*
- Test in VizDoom 3D simulator
- The results is similar with in Gibson
- Only feature-based agents can generalize without texture randomization during training


---
## Comments
- Does the mid-level perception contain "the fact that the world is 3D"?
- Is this no problems ? "realizing these gains requires careful selection of the mid-level perceptual skills". Curious about the explaination

---

## References
- [Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies](http://perceptual.actor/assets/main_paper.pdf)
