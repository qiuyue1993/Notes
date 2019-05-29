## Vision-based Navigation with Language-based Assistance via Imitation Learning with Indirect Intervention

### Indexing:
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Vision-based Navigation with Language-based Assistance](#Vision-based-Navigation-with-Language-based-Assistance)
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

#### Constraint formulation

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
