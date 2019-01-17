## Evaluating Visual Conversational Agents via Cooperative Human-AI Games

### Index
- [Introduction](#Introduction)
- [Conclusion](#Conclusion)
- [References](#References)
---
### Introduction
- Proposed GuessWhich game

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Evaluating-Human-AI-Team_overview.png" width="400" hegiht="400" align=center/>

- GuessWhich game interface

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Evaluating-Human-AI-Team_overview1.png" width="600" hegiht="400" align=center/>

- In this work, we design a cooperative game – GuessWhich – to measure human-AI team performance in the speciﬁc context of the AI being a visual conversational agent. 
- Experiments results show that  a mismatch between benchmarking of AI in isolation and in the context of human-AI teams.

Setting:
- The AI, which we call ALICE, is provided an image which is unseen by the human. 
-  Following a brief description of the image, the human questions ALICE about this secret image to identify it from a ﬁxed pool of images.
-  We measure performance of the human-ALICE team by the number of guesses it takes the human to correctly identify the secret image after a ﬁxed number of dialog rounds with ALICE.

---
### Conclusion
- Performance by MR(lower is better)

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Evaluating-Human-AI-Team_MR_results.png" width="400" hegiht="400" align=center/>

- Human perception ratings

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Evaluating-Human-AI-Team_Human_Perception_Ratings.png" width="400" hegiht="400" align=center/>

- Comparision of supervised learning (SL) and reinforcement learing (RL)

<img src="https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/images/Paper-Summarize_Evaluating-Human-AI-Team_Example-Comparision-Results.png" width="600" hegiht="400" align=center/>

- We ﬁnd that ALICERL (ﬁne-tuned with reinforcement learning) that has been found to be more accurate in AI literature than it’s supervised learning counterpart when evaluated via a questioner bot (QBOT)-ALICE team, is not more accurate when evaluated via a human-ALICE team. This suggests that there is a disconnect between between benchmarking of AI in isolation versus in the context of human-AI interaction.
- This suggests that while self-talk and RL are interesting directions to pursue for building better visual conversational agents,there appears to be a disconnect between AI-AI and human-AI evaluations. 
---
### References
- [Evaluating Visual Conversational Agents via Cooperative Human-AI Games](https://arxiv.org/pdf/1708.05122.pdf)
