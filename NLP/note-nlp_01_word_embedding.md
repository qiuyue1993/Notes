# The NLP Note - 01: Word Embedding

### Indexing:
- [The core of DeepNLP - Representation](#The core of DeepNLP - Representation)
- [Introduction](#introduction)
- Categories of NLP Representation 
-- One-Hot Representation
-- Distributed Representation
- NLP Language Model
- Distributed Representation
- Word Embeding
- Word2Vec
-- Neural Network  based Language Model
-- Word2Vec, CBOW, Skip-gram
- Neural Network Language Model and word2vec
-- Neural Network Language Model
-- Word2vec, CBOW, Skip-gram
-- Word Embedding
- Some thoughts
---
## The core of DeepNLP - Representation

- Top Conference: ACL
- Someone said that " *Image and Sound is kind of low level data* "
- However, language is highly abstract
-  **To use NN, the first biggest problem is how to represent language data**

---
### Categories of NLP Representation
#### One-hot Reprentation
- Eg. *Microwave* -> [0,0,1,0,0]
- Use one hot vector to represent a word, thus, for a vocabulary with 1,000 words, you need to use 1000-dim to represent every word.
- However, use one-hot vector + SVM, CRF, MEMM can do well in a variety of fields of NLP.
- **Drawback : requiring high dimensions, no relations between similiar words** 

#### Distributed Representation
- Distributional Hypothesis of Harris (1954), Firth(1957)  : *a word is characterized by the company it keeps*
- 3 types of distributed representation : 
-- Matrix based
-- Clustering based
-- Neural Network based

---
### NLP Language Model
- Statistics Language Model
-- A language (Sequence of Words) is a random event, and statistics language model use some probability to describe this event.
-- Define vocabulary set as $V$, a sequence (or sentence) $S = <w_1, w_2,...,w_m>\in V_n$, statistics language model computes a probability for this sequence $P(S)$ to measure the confidence of $S$ in conforming to the grammer and semantic rules.
- Popular Statistics Language Model
-- N-gram Model (unigram model, bigram model, trigram model)
-- Usually, $P(w_i\mid w_1, w_2, ..., w_{i-1})\approx P(w_i\mid w_{i-(n-1)}, ..., w_{i-1})$

---
### Distributed Representation
#### Matrix based Representation
- This also been called distributed semantic model
- A row of a matrix represent a word.
- The Euclid Distance of the vectors indicates the similarity of two words.
- Gloval Vector (GloVe) is a kind of matrix based representation.

#### Clustering based Representation
- Bypass

#### Neural Network based Representation
- Word Embedding is a kind of NN based representation.
- Modeling word representation by representing the context of target word.
- Merit : ability to represent conplicated context.

---
### Word Embedding
![avatar](C:/Users/qiu/Boostnote/notes/images/NLP_Note01/f1.png)
- Word Embedding can be considered as a by-product of the neural network language model.
- Comparing with one-hot vector representation, word embedding use dense representation (above figure). 
- For a word vector, every element is a real number.
- Absolutly, the dimension will be far smaller than one-hot vector.

---
- Neural Network Language Model and word2vec
-- Neural Network Language Model
-- Word2vec, CBOW, Skip-gram
-- Word Embedding

### Neural Network Language Model and word2vec
#### Neural Network Language Model
- Neural Network Language Model, NNLM
- Log-Bilinear Language Model, LBL
- Recurrent Neural Network based Language Model, RNNLM
- C&W Model
- CBOW (Continuous Bag of Words)
- Skip-gram

####  Word2vec, CBOW, Skip-gram
- word2vec
-- CBOW, Skip-gram are famous word2vec, can be used to train word vector
-- CBOW : Computing target word's probability according to the n- words before target words or n-words following it.
-- Skip-gram : Computing the probabilities of n- words before current words or n- words following it according to the current words.

- Eg : CBOW
-- "*I love you*" is our training data, and current target word is "*love*".
-- Define $c$ as the words number used to compute the target word.
-- Define one-hot vector for every word have $v$ dimension
-- Weight : $w_1 :v*n$, $w_2 :n*v$ dimension vectors.
-- Pooling operates : $maxpooling_{c\to1}$
-- Input is $c*v$ dimension matrix
-- CBOW can be denoted as : $maxpooling_{c\to1}( Input*w_1 )*w_2*$
-- Output : $1*v$ dimension vector, indicating the propability of each word
-- Training : Minimizing Cross Entropy or other losses functions.

- Un-, Weak- Supervised Learning or Supervised Learning
-- Un-, Weak- Supervised Learning : word2vec, auto-encoder
-- Supervised Learning usually have complicated structure and more precise.

---
### Some thoughts
- Word Embedding can capture the semantics of words and decine the embedding dimension.

___
### Reference
- [word embedding与word2vec: 入门词嵌入前的开胃菜 - 知乎](https://zhuanlan.zhihu.com/p/32590428)







