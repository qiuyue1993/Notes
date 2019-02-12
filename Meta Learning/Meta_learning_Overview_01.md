## Meta Learning Overview 01
### Indexing:
- [What is Meta Learning](#What-is-Meta-Learning)
- [Unsupervised Meta Learning](#Unsupervised-Meta-Learning)
- [Task-Agnostic Meta-Learning](#Task-Agnostic Meta-Learning)
- [References](#References)

---
### What is Meta Learning
#### Meta of Deep Learning
- Hyper Parameters: learning rate, batch size, input size, etc.
- Network Architecture
- Initialization
- Optimizer: SGD, Adam, RMSProp
- Network parameters: ??
- Loss Function
- Back-propagation

---
### Unsupervised Meta Learning
#### Unsupervised Learning Task
- Use unlabeled data to learn feature representation
- Apply learned feature representation to Few-Shot Learning 

#### How to update network parameters for unlabeled data
- Construct a network (Meta Network) to predict parameters to be updated

#### Unsupervised Learning via Meta-Learning
- Run unsupervised learning to learn a initial embedding function
- Cluster embedding multiple times
- Automatically construct tasks without supervision
- Run meta-learning on tasks

---
### Task-Agnostic Meta Learning


---
### References
- [最前沿：Meta Learning/Learning to Learn, 到底我们要学会学习什么？](https://zhuanlan.zhihu.com/p/32270990)
- [谈谈无监督Meta Learning的研究](https://zhuanlan.zhihu.com/p/46339823)
- [Meta Learning单排小教学](https://zhuanlan.zhihu.com/p/46059552)
- [任务无偏的元学习(Task-Agnostic Meta-Learning)：最小化任务性能之间的不平等](https://zhuanlan.zhihu.com/p/37076777)
