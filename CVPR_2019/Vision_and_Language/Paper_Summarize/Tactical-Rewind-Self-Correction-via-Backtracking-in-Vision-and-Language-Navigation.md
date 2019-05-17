## Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation

### Indexing:
- [Introduction](#Introduction)
- [Method](#Method)
- [Experiments](#Experiments)
- [Analysis](#Analysis)
- [Related Work](#Related-Work)
- [Conclusion](#Conclusion)
- [Questions](#Questions)
- [References](#References)

---
### Introduction
**Abstract**
- We present the **Frontier Aware Search with backTracking (FAST) Navigator**, a general framework for action decoding.
- We use **asynchronous search to boost any VLN navigator by enabling explicit backtrack when an agent detects if it is lost**

**Compare with former works**
- Current approaches **make local action decisions**
- Or score entire trajectories using beam search
- We **balance local and global signals** when exploring an unobserved environment; This lets us act greedily but use global signals to backtrack when necessary.

**Problems of former works**
- Exposure bias: model **cannot perform accurately given its self-generated sequence** when trained on ground-truth data
- Deficiency of student forcing models: once the agent has **deviated from the correct path**, the original instruction no longer applies, which causes the confused agent to **fall into loops** sometimes.
- Beam search, which collects multiple global trajectories to score, **runs counter to** the goal of building an **efficent** agent.

**Solutions**
- Our FAST model is a form a **asynchronous search**, which **combines global and local knowledge to score** and **compare partial trajectories of different lengths**.
- We evaluate our progress by modeling how closely our previous actions align with the given text instructions, the result determines **which local action to take** and **whether the agent should backtrack**

---
### Method
#### Settings
- Input: instruction $X$, contains sentences describing a desired trajectory.
- Input at each step $t$: surrounding $V_t^{k}$, $k$ denotes the viewponit number
- Action: $a_1, a_2, ...,a_T \in A$， use panoramic action space, including a *stop* action

#### Learning Signals
*Current works*
- Using only local information, e.g. greedy decoding
- Or fully sweeping multiple paths simultaneously, e.g. beam search

**LOGIT** $l_t$: 
- local distribution over action
- chosen local cation at time $t$ is denoted $l_t$
- Use one LSTM to encode the original instruction and another LSTM to decode the logits

**PM** $p_t^{pm}$:
- global progress monitor
- tracks how much an instruction has been completed
- Inputs: decoder LSTM's current cell state $c_t$, previous hidden state $h_{t-1}$, visual inputs $V_t$, attention over language embeddings $a_t$ to compute score $p_t^{pm}$ 

**SPEACKER** $S$:
- global scoring
- Given: a sequence of visual observations and actions
- Produce: train a seq2seq captioning model as a "speaker to produce a textual description.
- Benefits 1: automatically annotate new trajectories with the synthetic instructions
- Benefits 2: score the likelihood of given trajectory with original instruction

#### Framework
*Intention*
- Integrates the 3 signals $(l_t, p_t^{pm}, S)$
- Train new indicators to answer $Q_1, Q_2, Q_3, Q_4$

*Questions*
- Should we backtrack
- Where should we backtrack to
- Which visited node is most likely to be the goal
- When does it terminate this search

*Abstract*
- Do backtrack and compare partial trajectories
- For any partial trajectory, the last action is *proposed* and evaluated, but not *executed*
- The model chooses whether to expand a partial trajectory or *stop*
- Maintain two priority queques, a *frontier queque $Q_F$* for partial trajectories, a *global candidate queue Q_C* for completed trajectories
- FAST expands its optimal partial trajectory until it decides to backtrack (**Q_1**)

*Q_1: Should we backtrack?*
- Explore: backtracks to the most promising partial trajectory
- Exploit: commits to the current partial trajectory, always executing the best action 

*Q_2: Where should we backtrack to?*
- Two techniques to backtrack, each acting over the sequence of actions
- *Sum-of-log*: sums the log-probabilities of every previous action, thereby computing the probability of a partial trajectory
- *Sum-of-logits*: sums the unnormalized logits of previous actions

*Q_3: Which visited node is most likely to be the goal?*
- FAST considers every point visited as candidate
- Using global information to rank candidates.

*When do we terminate the search?*
- Two alternative stopping criteria:
- When a partial trajectory decides to terminate
- When expanded M nodes

#### Algorithm
- Frontier queue: at every step, adds all possible next actions to the frontier queue
- Candidate queue: at every step, adds its current location to the candidate queue
- Choose the best partial trajectory from the frontier queue under the local scoring function
- Perform the final action proposal
- Update the candidate queue, frontier queue
- Continue (exploit) or backtrack
- Repeat the process 

---
### Experiments
#### Dataset
- Room-to-Room (R2R) dataset

#### Evaluation Criteria
- TL (Trajectory Length): the average length of the navigation trajectory
- NE (Navigation Error): the shortest path distance between final location and the goal location
- SR (Success Rate): the percentage of the agent's final location is less than threshold to the goal
- SPL (Success weighted by Path Length): trades-off SR against TL

#### Baselines
- RANDOM
- SEQ2SEQ
- SPEAKER-FOLLOWER: an agent trained with data augmentation from a speaker model on the panoramic action space
- SMNA: agent trained with a visual-textual co-grounding module and a progress monitor

#### Our Model
- Two version
- FAST(short) uses the exploit strategy
- FAST(long) uses the explore strategy

#### Results
- In R2R dataset, FAST outperform previous approaches in almost every setting
- FAST can be intergrated with current approaches (SPEAKER-FOLLOWER, SMNA) easly

---
### Analysis
*Intention to isolate the effects of local and global knowledge, backtracking, stopping criteria*

#### Fixing Your Mistakes
- The proposed method increases the success rate at early step divergence from ground truth
- The greedy approach equally successful if it progresses over halfway through the instruction

#### Knowing When To Stop Exploring
- We investigated the number of nodes to expand before terminating the algorithm.
- The model's success rate, though increasing with more nodes expands 40 nodes, does not match the oracle's rate

#### Local and Global Scoring
*Core to our apprach*
- frontier queue for expansion
- candidate queue for proposing the final candidate

*Fusion methods for scoring partial trajectories*
*Fusion methods for ranking complete trajectories*

#### Intuitive Behavior
- The greedy decoder is forced into a behavioral loop because only local improvements are considered.
- Using FAST clearly shows that even a single backtracking step can free the agent of poor behavioral choices.

---
### Related Work
- Due to the enormous visual complexity of real-world scenes, the VLN literature usually builds on computer vision work from referring expressions, visual question answering, and ego-centric QA that requires navigation to answer questions.

---
### Conclusion
- We present FAST NAVIGATOR, a framework for using asynchronous search to boost any VLN navigator by enabling explicit backtrack when an agent detects if it is lost.

---
### Questions
- Why **LOGIT** $l_t$ don't take visual information as one of the inputs
- Why **PM** $p_t^{pm}$ don't take language information as one of the inputs

---
### References
- [Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation](https://arxiv.org/pdf/1903.02547.pdf)
- [Video](https://www.youtube.com/watch?v=AD9TNohXoPA&feature=youtu.be)
- [Code](https://github.com/Kelym/FAST)
---
