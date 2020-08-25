# Reading list

## TextCaps: a Dataset for Image Captioning with Reading Comprehension

Existing problem: Current approaches are unable to include written text in their descriptions, although text is omnipresent in human environments and frequently critical to understand our surroundings. 

Propose dataset: To study how to comprehend text in the context of an image we collect a novel dataset, TextCaps, with 145k captions for 28k images. Our dataset challenges a model to recognize text, relate it to its visual context, and decide what part of the text to copy or paraphrase, requiring spatial, semantic, and visual reasoning between multiple text tokens and visual entities, such as objects.

Our analysis also points out several challenges of this dataset: Different from other captioning datasets, nearly all our captions require integration of OCR tokens, many are unseen (“zero-shot”). In contrast to TextVQA datasets, TextCaps requires generating long sentences and involves new technical challenges, including many switches between OCR and vocabulary tokens.

Experiment:  We study baselines and adapt existing approaches to this new task, which we refer to as image captioning with reading comprehension. Our analysis with automatic and human studies shows that our new TextCaps dataset provides many new technical challenges over previous datasets.


## Object-and-Action Aware Model for Visual Language Navigation

Existing problems: most existing methods pay few attention to distinguish these information from each other during instruction encoding and mix together the matching between textual object/action encoding and visual perception/orientation features of candidate viewpoints.

Proposed method: We propose an Object-and-Action Aware Model (OAAM) that processes these two different forms of natural language based instruction separately. 

However, one side-issue caused by above solution is that an object mentioned in instructions may be observed in the direction of two or more candidate viewpoints, thus the OAAM may not predict the viewpoint on the shortest path as the next action. To handle this problem, we design a simple but effective path loss to penalize trajectories deviating from the ground truth path. To encourage the robot agent to stay on the path, we additional propose a path loss based on the distance to nearest ground-truth viewpoint.

Result: Experimental results demonstrate the effectiveness of the proposed model and path loss, and the superiority of their combination with a 50% SPL score on the R2R dataset and a 40% CLS score on the R4R dataset in unseen environments, outperforming the previous state-of-the-art.

## 






