## Toxic Content Monitoring Notebooks Folder
The collected notebooks pertaining to model training and testing.

### **Guide:**

### Section 1

BalancedRandomForest:
* 0.1-balanced-random-forest-tfidf.ipnb  - TFIDF vectorizer.
* 0.1-balanced-random-forest-elmo.ipnb  - Elmo encodings.
* 0.1-balanced-random-forest-bert.ipnb  - Bert encodings.
* 0.1-balanced-random-forest-bert-small.ipnb - Small Bert encodings.

DecisionTrees:
* 0.1-decision-tree-tfidf.ipynb.ipynb - TFIDF vectorizer.
* 0.1-decision-tree-elmo.ipynb - Elmo encodings.
* 0.1-decision-tree-bert.ipynb - Bert encodings.
* 0.1-decision-tree-bert-small.ipynb - Small Bert encodings.

Logistic Regressions:
* 0.1-logistic-regression-with-tfidf.ipynb - TFIDF vectorizer.
* 0.2-logistic-regression-with-tfidf.ipynb - TFIDF vectorizer.
* 0.1-logistic-regression-with-elmo.ipynb - Elmo encodings.
* 0.1-logistic-regression-with-bert-encodings.ipynb - Bert encodings.
* 0.1-logistic-regression-with-small-bert-encodings.ipynb - Small Bert encodings.
* 0.1-logistic-regression-with-basilica-encodings.ipynb - Basilica encodings.

RUSBoost:
* 0.1-rusboost-classifier-with-tfidf.ipynb - TFIDF vectorizer.
* 0.1-rusboost-classifier-with-elmo.ipynb - Elmo encodings.
* 0.1-rusboost-classifier-with-bert.ipynb - Bert encodings.
* 0.1-rusboost-classifier-with-small-bert.ipynb - Small Bert encodings.

GloVe:
*  0.1-glove-300.ipynb - GloVe

### Section 2

* 0.1-toxic-tentament.ipynb - A working notebook which contains both the Vader analysis as well as the custom classifier made in NLTK.
* 0.1-top-one-hundred-toxix-words.ipynb - Find top 100 words associated with true examples of each classification problem.
* 0.1toxic-rnn.ipynb - notebook for creating an RNN encoder (Seq2Vec) model. I suspect this suffers from over-fitting. Used to create `toxic_rnn.pt` and `toxic_rnn_matrix.out`.
* 0.1-kaggle-word-splits.ipynb - notebook for text processing based on kaggle conpetition best notebook.
* 0.1-hf-bert.ipynb - Used to make encoding set for BERT using the small dataset + the huggingface transformers library.
* 0.1-allen-nlp-bert.ipynb - Used to make a BERT encoding using the AllenNLP library.
* 0.1-amazon-translation.ipynb - Contains code used to mass-translate russian using Amazon's API.
* 0.1-basilica-embeddings.ipynb - Notebook that encodes toxic-train-clean.csv using
Basilica and also basilica_y.csv.