# Lecture 1

## Background
- The core of reinforcement learning is **decision making**
- A sequential decision making problem

## Characteristics of Reinforcement Learning
- There is no supervisor, only a **reward signal**
- Feedback is delayed, not instantaneous
- Time really matters (sequential, non i.i.d data)
- Agent's actions affect the subsequent data it receives

## A challenge in reinforcement learning
- One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the **trade-off between exploration and exploitation**
- Another key feature of reinforcement learning is that is explicitly considers the whole problem of a goal-directed agent interacting with uncertain environment

## Examples of Reinforcement Learning
- Fly stunt manoeuvres in a helicopter
- Defeat the world champion at Backgammon
- Manage an investment portfolio
- Control a power station
- Make a humanoid robot walk
- Play many different Atari games better than humans

## Rewards
- A reward $R_t$ is a scalar feedback signal
- Indicates how well agent is doing at step $t$
- The agent's job is to **maximise cumulative reward** 

## Sequential Decision Making
- Select actions to maximise total future reward
- Actions may have long term consequences
- Reward may be delayed
- It may be better to sacrifice immediate reward to gain more long-term reward

## Agent and Environment
*Agent*
- At each step $t$, executes action $A_t$
- Receives observation $O_t$
- Receives scalar reward $R_t$

*Environment*
- Receives action $A_t$
- Emits observation $O_{t+1}$
- Emits scalar reward $R_{t+1}$

## History and State
- $H_t = O_1, R_1, A_1, ..., O_{t-1}, R_{t-1}, A_{t-1}, O_t, R_t, A_t$
- **State** is the information used to determine what happens next
- State: $S_t = f(H_t)$

*Environment State*
- The environment's private representation
- Contain all data to decide next observation / reward

*Agent State*
- The agent's internal representation
- The information used by reinforcement learning algorithm

*Information State*
- Contains all useful information from the history
- A state $S_t$ is Markov is and only if:

$$
P[S_{t+1} | S_t] = P[S_{t+1} | S_1, ..., S_t]
$$

- **The future is independent of the past given the present**

## Fully Observation Environment
- $O_t = s_t^{e} = s_t^{a}$
- Formally, this is a **Markov decision process (MDP)**

## Partially Observable Environments
- $O_t \neq S_t^{a}$
- Agent must construct its own state representation 

## Major Components of an RL Agent
- Policy: **agent's behaviour function**
- Value function: how good is each state and /or action
- Model: agent's representation of the environment

### Policy
- The agent's behaviour
- A map from state to action
- Deterministic policy: a = \pi (s)
- Stochastic policy: \pi (a|s) = P[A_t = a | S_t = s]

### Value Function
- A prediction of future reward

### Model
- Predicts what the environment will do next
- $P$ predicts the next state
- $R$ predicts the next reward

## Categorizing RL agents (1)
*Value Based*
- Value Function
- No Policy (Implicit)

*Policy Based*
- Policy
- No Value Function

*Actor Critic*
- Policy 
- Value Function

## Categorizing RL agents (2)
*Model Free*
- Policy and / or Value Function
- No Model

*Model Based*
- Policy and / or Value Function
- Model

## Learning and Planning
*Reinforcement Learning*
- The environment is initially unknown
- The agent interacts with the environment
- The agent improves its policy

*Planning*
- A model of the environment is known

## Exploration and Exploitation
- *Exploration* finds more information about the environment
- *Exploitation* exploits known information to maximise reward

## Prediction and Control
- **Prediction: evaluate the future** ??
- **Control: optimise the future** ??

