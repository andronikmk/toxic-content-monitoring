import nni
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import logging
import scipy
import numpy as np
import pandas as pd

LOG = logging.getLogger('logistic_regression_tfidf_classification')

def load_data():
    '''Load dataset, use toxic elmo dataset'''
    X = np.loadtxt('toxic_elmo_matrix.out', delimiter=',')
    y = pd.read_csv("train-ys.csv").values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
          'estimator__C': 1,
          'estimator__penalty': "l2"
          }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', n_jobs=-1))
    model.estimator.C = PARAMS.get('estimator__C')
    model.estimator.penalty = PARAMS.get('estimator__penalty')

    return model

def run(X_train, X_test, y_train, y_test, model):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    testProba = model.predict_proba(X_test)
    score = roc_auc_score(y_test, testProba)
    LOG.debug('ROC-AUC score: %s' % score)
    nni.report_final_result(score)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
