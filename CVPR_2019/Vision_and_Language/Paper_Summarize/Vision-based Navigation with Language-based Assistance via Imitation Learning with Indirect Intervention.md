## Vision-based Navigation with Language-based Assistance via Imitation Learning with Indirect Intervention

### Indexing:
- [Introduction](#Introduction)
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
### Future Work
- Provide more natural, **fully linguistic question and answer interactions** between advisor and agent.
- Investigating how to transfer **from simulators to real-world robots**.

---
### References
- [Vision-based Navigation with Language-based Assistance via Imitation Learning with Indirect Intervention](https://arxiv.org/pdf/1812.04155.pdf)
- [Code](https://github.com/debadeepta/vnla)
- [Video Demo](https://www.youtube.com/watch?v=Vp6C29qTKQ0&feature=youtu.be)

---
