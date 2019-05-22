## Visual Storytelling

### Indexing:
- [Instruction](#Instruction)
- [Motivation and Related Work](#Motivation-and-Related-Work)
- [Dataset Construction](#Dataset-Construction)
- [Data Analysis](#Data-Analysis)
- [Automatic Evaluation Metric](#Automatic-Evaluation-Metric)
- [Baseline Experiments](#Baseline-Experiments)
- [Conclusion and Future Work](#Conclusion-and-Future-Work)
- [References](#References)

---
### Instruction
**Abstract**
- Introduced the first dataset for **sequential vision-to-language**.
- Proposed the task of visual storytelling.

**Dataset Abstract SIND v1.0**
- 81,743 unique photos
- 20,211 sequences
- DII: Descriptions of Images-in-Isolation
- DIS: Descriptions of Images-in-Sequence
- SIS: Stories  of Images-in-Sequence

---
### Motivation and Related Work
- Related Work: Image Captioning; Question Answering; Visual Phrases; Video Understanding; Visual Concepts
- Storytelling itself is one of the oldest known human activities, providing a way to educate, preserve culture, instill morals, and share
advice.
---
### Dataset Construction
#### Extracting Photos
- Albums with **EVENT**
- Only include albums **with 10 to 50 photos**, where all photos are taken **within a 48-hour span**
- Using AMT to collect the corresponding stories and descriptions

#### Crowdsourcing Stories In Sequence
- 2-stage crowdsourcing workflow
- First stage: **storytelling**
- Select a subset of photos from a given album
- Writes a story about it
- Second stage: **re-telling**
- Writes a story about it

#### Crowdsourcing Descriptions of Images In Isolation & Images In Sequence
- Image captioning for Images and Sequences of former step

#### Data Post-processing
- Tokenize all storylets and descriptions with the **CoreNLP tokenizer**
- Replace people names with generic **MALE/FEMALE** tokens
- Identified named entities with their entity type (e.g., *location*)
- training, validation, test split for 80%, 10%, and 10%

---
### Data Analysis
- 10,117 Flickr albums with 210,819 unique photos
- average time span of 7.9 hours
- Identify the words most closely associated with each tier, results show that
- DII: abundant use of posture verbs
- DIS: relatively uninformative words are much less represented
- SIS: include more storytelling elements, such as names, temporal references, and words more dynamic and abstract

---
### Automatic Evaluation Metric
- Compute **pairwise correlation coefficients between automatic metrics and human judgements**
- Human judgements: AMT
- Automatic metrics: METEOR, smoothed-BLEU, Skip-Thoughts
- **METEOR** correlates best with human judgement

---
### Baseline Experiments
- Sequence-to-sequence recurrent neural network
- Image encoder: GRUs
- Story decoder: GRUs

**Decoder Design**
- beam search with lower decoder beam size
- greedy search: results many repeated words and phrases
- better greedy search: force the same content word cannot be produced more than once within a given story
- Use the words of image caption

---
### Conclusion and Future Work
- First dataset for **sequential vision-to-language**
- Established several strong baselines
- METEOR as an automatic metric
---
### References
- [Visual Storytelling](https://aclweb.org/anthology/N16-1147)
---
