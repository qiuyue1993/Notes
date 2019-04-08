## Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding

### Indexing
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Approach](#Approach)
- [Evaluations](#Evaluations)
- [Discussion](#Discussion)
- [References](#References)
---
### Introduction
**Abstract**:
- Our neural-symbolic VQA (NS-VQA) system first recovers a **structural scene representation** from the image and **program trace** from
the question. It then **excutes the program on the scene representation** to obtain answer.

**Advantages of incorporating symbolic structure as prior knowledge**:
- Robust to long program traces, thus can solve **complex reasoning** better.
- More data- and memory-**efficient**.
- Offering full **transparency** to the reasoning process.

**Comparison with former work**:
- Our neural-symbolic approach **fully disentangles vision and language understanding from reasoning**.

---
### Related Work
#### Structural scene representation
- [Convolutional inverse graphics networks](https://arxiv.org/pdf/1503.03167.pdf)
- [Learning disentangled representations of pose and content](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf)
- [Disentangled representation without direct supervision](https://arxiv.org/pdf/1707.03389.pdf)
- [Vision as inverse graphics](https://pdf.sciencedirectassets.com/271877/1-s2.0-S1364661306X01195/1-s2.0-S1364661306001264/main.pdf?x-amz-security-token=AgoJb3JpZ2luX2VjEHgaCXVzLWVhc3QtMSJIMEYCIQD6UvP49dRiZ1W7GwYG%2BPqfyd4CccOFjKlCaT72dFVJPwIhAPhjpkm5YuayjNTNAKvKrut1IHRtV5rs2gj0oL%2F2aTuyKtoDCEEQAhoMMDU5MDAzNTQ2ODY1IgwTWtEXnHpmspahjkoqtwOP8ho5x55CspuMv96yl056khiNgIGMDx8vGMFarYjUuqr%2BLxtARigTdsGGB5ZX08IGmMZZsH%2BZA8n3kWQC8HqYxByXpR1s65TTdD52VtRT%2BBQpxa1y0t0KGK4LixTb%2FR5QqgUWRdgeVeZGoAy3Bjwr9i9JdQgh7AhmLxQzS1Gqrpl4YMc4fVH9zk9g7RjqTJCzUxJb5mMXPsj0ICoyNvnjn9xK%2B%2F%2BpZqaJy2xsaNEwxT5Efe9%2BlROGSjzo4FLKw80SQZC09OgFcEH1Y0sydue93cf8rBVXPdtz0tiPIpszr%2FpzzNRb%2BIEJ%2B7w7YjD5efiCvD5DhsZ9b6WOQwE2mSnRHWGACkv%2FdAnDsa5GXe1iZA94tFCmF0whXpGUDSUJmi1VaGJ3qEkQB1N7vEoUN9jqjNY0Oxtz4KpsIrbZ4sPm7synAQJZM9j%2BzpOtka7xun92B%2BrsKtMjRhzSkp1OGee2tb4lER%2BK4T8Ekpa%2FdGG3pvdU541ECQ00yzSacN%2Fey2Xaj%2BOubh8zQ9lnrWoUwjeFNBl%2BfbANwhWgeg9hEcD81vzrirm5wano8tNxi%2FqG5j9nCmVqS6AxMJzvq%2BUFOrMBttx560dHw%2F5hWCXDAjmzObeOEZXvF1ZSzjTBmezYsLOwPTIs9f0mMJrU%2B5P%2BoeQ3q5YdVR9mDwi2EEju4chrWbUhI9ArSlw7VyORxDaY%2FEvCxdDbTMh%2Fjt6gPiEp%2Bd0vjtZBBjk724jC1GeHG7xAdvpCXoomZ9krx5TIe%2BSkggwU%2F9hhcLu6W8y3a3QzHMBDbxAlWp392SoAEQwSibQ5TSZmCGfQdVf4v%2BCBzevAMLWI178%3D&AWSAccessKeyId=ASIAQ3PHCVTYUBQTLTG2&Expires=1554711805&Signature=Sg5gHBE%2FMo%2BMsdpC%2FMXEY4mdnXM%3D&hash=e72336ee7d1f2189e8787249349224180dcfba4c7e98d370cac712dc180f716f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1364661306001264&tid=spdf-bddd8b6b-0867-41cc-a229-ffba38490831&sid=8e287f9b8c8b474ce689771-ed1ba3d2a6c6gxrqb&type=client)

#### Program induction from language
- Using program search and neural networks to recover programs from a domain-specific language
- Semantic parsing methods map sentences to logical forms via a knowledge base or a program
- Use latent structure in language to help question answering and reasoning

#### Visual question answering
- Works explicitly use structure knowledge to help reasoning
- Approaches based on nerual attentions
- Works that reveal the fundamental problem of VQA system-models are overfitting dataset biases.

#### Visual Reasoning
- CLEVR
- [Compositional Attention Networks for Machine Reasoning](https://cs.stanford.edu/people/dorarad/mac/blog.html)
- [Relation Network for VQA](https://arxiv.org/pdf/1706.01427.pdf)
- [Transparency by Design](https://arxiv.org/pdf/1803.05268.pdf)
- [VQA on General Dependency Tree](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Visual_Question_Reasoning_CVPR_2018_paper.pdf)
- [Learning Interpretable Spatial Operations](https://arxiv.org/pdf/1712.03463.pdf)

---
### Approach
**Overview**
- A scene parser (de-renderer): de-render image to obtain a **structural scene representation**
- A question parser (program generator): generate a hierarchical program from the question
- A program executor: run the program on the structural representation to obtain an answer

**Scene parser**
- Step 1: use Mask R-CNN to generate **segment proposals** of all objects. Also predicts the **categorical labels** of discrete instrinct attributes. (0.9 threshold for proposals)
- Step 2: extract spatial attributes (pose, 3D coordinates) from object proposals and original image

**Question parser**
- Attention-based sequence to sequence model with encoder-decoder structure.
- Encoder: bidirectional LSTM
- Decoder: LSTM

**Program executor**
- A collection of deterministic, generic functional modules in Python.
- Each functional module sequentially executes on the output of previous one.
- The last module outputs the final answer to the question.

**Traning Paradigm**
- Scene parsing: Mask R-CNN (ResNet-50 FPN as backbone, train the model for 30,000 iterations with 8 image mini-batch), train proposed object segments computed from the training data for 30,000 for object categorical labels prediction. Both networks are trained on 4,000 generated CLEVR images.
- Reasoning: Supervised training with 20,000 iterations, **reinforce training** for 2M iterations.
---
### Evaluations
#### Data-Efficient, Interpretable Reasoning
- Setup: CLEVR dataset; Ablation of numbers of ground-truth programs; Compared with IEP baseline.
- Results: Outperforms all of the other methods on all five question types.
- High data-efficiency and recovering of underlying programs.

#### Generalizing to Unseen Attribute Combinations
- Setup: CLEVR-CoGenT dataset
- Results: Generalize well after finetuning.

#### Generalizing to Questions from Humans
- Setup: CLEVR-Humans dataset
- Results: Outperforms IEP by a considerable margin.

#### Extending to New Scene Context
- Setup: A new dataset with images from Minecraft; With 10,000 Minecraft scenes and 100K questions.
- Results: behavior is similar to CLEVR.

---
### Discussion
- We have presented a **neural-symbolic VQA approach** that disentangles reasoning from visual perception and language understanding.
- Intergrating **unsupervised or weakly supervised representation learning** in both language and vision with our neural-symbolic approach to **visually grounded language** is a promising future direction.

---
### References
- [Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/pdf/1810.02338.pdf)
