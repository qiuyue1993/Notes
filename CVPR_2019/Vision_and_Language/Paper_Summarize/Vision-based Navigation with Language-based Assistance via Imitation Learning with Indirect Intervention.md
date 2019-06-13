## Vision-based Navigation with Language-based Assistance via Imitation Learning with Indirect Intervention

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Vision-based Navigation with Language-based Assistance](#Vision-based-Navigation-with-Language-based-Assistance)
- [Imitation Learning with Indirect Intervention](#Imitation-Learning-with-Indirect-Intervention)
- [Environment and Data](#Environment-and-Data)
- [Implementation](#Implementation) 
- [Experimental Setup](#Experimental-Setup)
- [Results](#Results)
- [Future Work](#Future-Work)
- [References](#References)

---
### Introduction
**Tasks**
- The requester may not know how to navigate to the target objects and thus **makes requests by only specifying high-level endgoals**.
- The agent is **capable of sensing when it is lost** and querying an advisor.

**Task details**
- Request tasks through **high-level** instructions that only describe **end-goals**
- Introduce an **advisor** to provide the agent with **low-level language subgoals**

**Steps**
- Ground the object referred with th initial end-goal in raw visual inputs
- Sense when it is lost and use an assigned budget for requesting help
- Execute language subgoals to make progress

**I3L**

*Different from conventional Imitation Learning*

- Advisor present in environment not only during training time but also test time
- Assist by modifying the environment instead of directly making decisions

---
### Related Work
#### Language and robots
- **Learning-from-Demonstration (LfD)**: ground natural language to robotic manipulator instructions using LfD
- Research that employing **imitation learning** of natural language instruction using human following directions as demonstraction data
- Research using verbal constraints for safe robot navigation in complex real-world environments

#### Simulated environments
- Puddle World Navigation
- Rich 3D Blocks World
- Current simulators are more photo-realistic, and beginning to be utilized to train real world embodied agents

#### End-to-end learning in rich simulators
- Embodied Question Answering
- Vision-Language navigation task: cross-modal matching and self-learning significantly improve generalizability to unseen environments

#### Imitation learning
- Task that teacher is available or can be simulated
- Studies where teachers provide imperfect demonstrations
- Studies that construct policies to minimize the number of queries

---
### Vision-based Navigation with Language-based Assistance
#### Setup
- The agent starts at a **random location**
- A requester assigns it an **object-finding task** by sending a high-level end-goal
- The agent is considered to success if it stops at a location within *d* (**task-specific hyperparameter**) meters along the shortest path to an instance of desired object.
- The agent can **sense** if it **get lost**
- The advisor then repsonds with language providing a **subgoal**, subgoal **describe the next *k* optimal actions**

#### Constraint formulation
- The agent faces a **multi-objective problem**: maxmizing success rate, minimizing hep request
- **Solution**: hard-constrained formulation-**maximizing success rate without exceeding a budgeted number of help requests**

---
### Imitation Learning with Indirect Intervention
- **I3L**: models scenarios where a learning agent is **monitored by a more qualified expert** and **receives** help through an **imperfect communication channel** (e.g., language)

#### Advisor
*Conventional Imitation Learning (IL)*
- Interaction between a learning agent and a teacher
- The agent learns by querying and imitation demonstrations of the teacher

*I3L*
- Agent also receives guidance from an advisor
- Teacher: only interacts with the agent at training time
- Advisor: assist the agent during both training and test time

#### Intervention
*Intervention*
- The advisor directs the agent to take a sequence of actions through an intervention

*Direct Intervention*
- The advisor overwrites the agent's decisions with its own
- The agent always takes actions the advisor wants it to take

*Indirect Intervention*
- The advisor does not take over the agent, but instead modify the environment to influence its decisons
- To utilize indirect interventions, the agent must learn to **interpret the signals in the environment to sequences of actions**
- Introduce an new error: **intervention interpretation error**

#### Formulation
- The environment is a **Markov decision process** with state transition function
- Policy 1 for agent: main policy for making decisions on the main task
- Policy 2 for agent: help-requesting policy for deciding when the advisor should intervene
- Teacher policies: main, help
- Advisor policy: phai
- Techer policies are only available during training, the advisor is always present

*Summarize*
- When the intervention instructs the agent to take the next *k* actions suggested by the teacher
- The state distribution induced by the agent, depends on both policy main and help
- The agent's objective is to minimize expected loss on the agent-induced state distribution

#### Learning to Interpret Indirect Intervensions
- I3L can be viewed as an imitation learning problem in a dynamic enviroment

*An I3L problem*
- can be **decomposed into a series of IL** problems; each of which can be solved with standard IL algorithms
- Teacher navigation policy: optmial shortest-path teacher navigation policy
- Follow subgoal while made mistakes or follow the teacher all the time are not good
- **solution**: BCUI (Behavior Cloning Under Interventions), to mix IL with behavior cloning
- The agent uses the teacher policy as the acting policy (behavior cloning) when executing an intervention
- As a result, the agent never deviates from the trajectory suggested by teh intervention

#### Connection to imitation learning and behavior cloning
*Behavior Cloning*
- Training time: advisor always intervenes
- Test time: agent make decision on its own

*IL*
- Training time: advisor intervenes randomly
- Test time: agent make decision on it own

*I3L-BCUI*
- Training time: learned policy decides when advisor intervenes, environment changed clue to interventions
- Test time: agent makes decisions on its won, learned policy decides when advisor intervenes, environment changed clue interventions 
- The advisor in I3L-BCUI intervenes both directly (through behavior cloning) and indirectly (by modifying the environment) at training time, buit only indirectly at test time

---
### Environment and Data
#### Matterport3D simulator
- **Matterport3D dataset**: large RGB-D dataset for scene understanding in indoor environments
- 90 real building-scale scenes
- 10,800 panoramic views
- 194,400 RGB-D images
- Residential building consisting of multiple rooms, floor levels, annotated with surface construction, camera poses, and semantic segmentation
- Former methods implemented a simulator that **the pose of the agent is specified by its viewpoint and orientation (heading angle and elevation angle)
- Edges connect reachable panoramic viewpoints that are less than 5m apart.

#### Visual input
- Simulator generates an RGB image representing the current first-person view of agent
- The image is fed into a ResNet-152 pretrained on ImageNet

#### Action space
- State-independent action space: left, right, up, down, forward, stop
- Help-requesting action space: request, do_nothing

#### Data Generation
- Based on the **Matterport3D dataset**, created a dataset **ASKNAV**
- 61 training, 11 development, 18 test
- Define data point as a tuple: environment, start pose, goal viewpoints, end-goal
- End-goal: Find [O] in [R]
- Development and test sets: seen set and unseen set

---
### Implementation
#### Notation
- The agent maintains two policies: a navigation policy and help-requesting policy.
- Each policy is stochastic, outputting a distribution **p** over its action space.
- An action *a* is chose by selecting the maximum probability action or sampling from the output distribution.
- The agent is supervised by a navigation teacher and a help-requesting teacher, and is assisted by an advisor

*Dataset*
- *d-th* data point consists of a start viewpoint
- a start orientation
- a set of goal viewpoints
- a end goal
- the full map

#### Algorithm
- Train navigation policy: I3L-BCUI algorithm
- Train help-requesting policy: behavior cloning

At time step *t*:
- Agent receives a view of the environment
- Agent computes a **tentative navigation distribution**
- The tentative navigation distribution is used as an input to **compute a help-requesting distribution**
- The agent **invokes the help-requesting teacher** to decide if it should request help
- Either the advisor is invoked to provide help, using the subgoal to be the end-goal
- Or the requesting help is not met, the end-goal keep unchange
- When the last-requesting action is executed, the agent selects the acting navigation policy based on the principle of the I3L-BCUI
- When the request is within the last *k* steps, the teacher policy acts

#### Agent
- Separate nn modules for Navigation policy and Help-requesting policy
- **Navigation module**: encoder-decoder model with a multiplicative attention mechanism and converage modeling, encode an end-goal, decodes a sequence of actions
- **Help-requesting module**: a multi-layer feed-forward nn with ReLU activation functions and a softmax final layer

#### Teachers
*Navigation teacher*
- Choose actions to traverse along the shortest path from the current viewpoint to the goal viewpoints
- Issues the *stop* action when one of the goal viewpoints is reached

*Help-requesting teacher*
- Time to request help 1: The agent deviates from the shortest path distant than a threshold distance
- Time to request help 2: The agent is "confused". (Decided by the entropy)
- Time to request help 3: The agent has remained at the same viewpoint for a fixed steps
- The help-request budget is greater than the number of remaining steps
- The agent is at a goal viewpoint but the highest navigation distribution is *forward*

#### Advisor
- The advisor queries the navigation teacher for *k* consecutive steps
- Then he actions are aggregated to make the language

#### Help-request Budget
- Define a hyperparameter which is the ratio between the total number of steps where the agent receives assistance and the time budget

---
### Experimental Setup
#### Baselines
- LEARNED: learned help-requesting policy
- NONE: never request help
- FIRST: request help continuously till a fixed steps
- RANDOM: uniformly randomly choose a fixed steps to request help
- TEACHER: follows the help-requesting teacher

#### Evaluation metrics:
- success rate
- room-finding success rate
- navigation error


---
### Results
#### Main Results
- Requesting help is more useful in unseen environments
- LEARNED policy outperforms all agent-agnostic polices
- There is a gap between LEARNED and TEACHER
- RANDOM is better than FIRST for unseen environments, for seen environments, the results are opposite

#### Effects of subgoals
- Receiving subgoals boosts sucess rate on TEST UNSEEN whether interention is direct or indirect

#### Does the agent learn to identify objects?
- The agent which is equipped with a learned help-requesting policy and trained with room types, learns to recognize objects.

---
### Future Work
- Provide more natural, **fully linguistic question and answer interactions** between advisor and agent.
- Investigating how to transfer **from simulators to real-world robots**.

---
### References
- [Vision-based Navigation with Language-based Assistance via Imitation Learning with Indirect Intervention](https://arxiv.org/pdf/1812.04155.pdf)
- [Code](https://github.com/debadeepta/vnla)
- [Video Demo](https://www.youtube.com/watch?v=Vp6C29qTKQ0&feature=youtu.be)

---
