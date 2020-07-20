## Toxic Content Monitoring Models Folder
The files pertaining to hyperparameter tuning and neural architecture search through the [NNI](https://github.com/microsoft/nni) library.

### **Guide:**

* Pipfile - contains the pip environment used for hyperparameter tuning with sklearn.
* notes.txt - contains some notes pertaining to usage.
* RNN NAS - Contains a pytorch NAS attempt.
* RNN Hyperparameter Tuning - Contains a pytorch NN hyperparameter tuner, as well as a notebook for generically training the network.
* Elmo Logistic Regression - NNI Hyperparameter search example, Logistic Regression trained on Elmo outputs.
* Elmo Random Forests - NNI Hyperparameter search example, Random Forests trained on Elmo outputs.
* TFIDF Logistic Regression - NNI Hyperparameter search example, Logistic Regression trained on TFIDF outputs.
* TFIDF Random Forests - NNI Hyperparameter search example, Random Forests trained on TFIDF outputs.

**Note:** NNI cannot be installed simultaneously with imbalanced-learn through pipenv. Instead, one must install it separately through pip, which doesn't complain.
