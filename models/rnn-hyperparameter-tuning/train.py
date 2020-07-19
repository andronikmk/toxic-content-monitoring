import nni

import logging

import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from torchnlp.word_to_vector import FastText

from torch.nn.utils.rnn import pad_sequence

LOG = logging.getLogger('rnn_hyperparameter_tuning')

vectors = FastText()
device = torch.device("cuda:0")

BATCH_SIZE = 96

def load_data():
    '''Load dataset, data cleaned using Kaggle method.'''
    df = pd.read_csv("toxic-train-kaggle-clean.csv")
    df["word_splits"] = df["word_splits"].apply(eval)
    df = df[(df["word_splits"].apply(len) > 0) & (df["word_splits"].apply(len) <= 560)]

    X_train, X_test, y_train, y_test = train_test_split(df["word_splits"], df.drop("word_splits", axis=1), random_state=99, test_size=0.15)

    X_train = X_train.values
    y_train = y_train.values

    X_test = X_test.values
    y_test = y_test.values

    batched_X_train = []
    batched_y_train = []

    i=0
    while (i+1) * BATCH_SIZE < len(X_train):
        batched_X_train.append(X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        batched_y_train.append(y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        i+=1
    batched_X_train.append(X_train[i*BATCH_SIZE:])
    batched_y_train.append(y_train[i*BATCH_SIZE:])

    batched_X_test = []
    batched_y_test = []

    del X_train
    del y_train

    i=0
    while (i+1) * BATCH_SIZE < len(X_test):
        batched_X_test.append(X_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        batched_y_test.append(y_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        i+=1
    batched_X_test.append(X_test[i*BATCH_SIZE:])
    batched_y_test.append(y_test[i*BATCH_SIZE:])

    del X_test
    del y_test

    return batched_X_train, batched_y_train, batched_X_test, batched_y_test

class ToxicClassifierModel(nn.Module):
    def __init__(self,
                 LSTM_UNITS = 128,
                 dropout_rate = 0.2,
                 hidden1Activation = F.relu,
                 hidden2Activation = F.relu,
                 #hidden1Size = 512,
                 #hidden2Size = 512
                 ):
        super(ToxicClassifierModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.BiGRU = nn.GRU(300, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=1)
        self.BiRNN = nn.RNN(input_size = 2 * LSTM_UNITS, hidden_size = LSTM_UNITS, bidirectional=True)
        self.hidden1 = nn.Linear(4 * LSTM_UNITS, 4 * LSTM_UNITS)
        self.hidden1Activation = hidden1Activation
        self.hidden2 = nn.Linear(4 * LSTM_UNITS, 4 * LSTM_UNITS)
        self.hidden2Activation = hidden2Activation
        self.hidden3 = nn.Linear(4 * LSTM_UNITS, 6)
    
    def forward(self, X):
        X = X.permute(0, 2, 1)
        X = F.dropout2d(X, self.dropout_rate, training=self.training)
        X = X.permute(0, 2, 1)
        
        X = self.BiGRU(X)
        
        X = self.BiRNN(X[0])
        
        X = X[0]
        
        X = torch.cat((torch.max(X, 1).values, torch.mean(X, 1)), 1)
        
        X = X.add(self.hidden1Activation(self.hidden1(X)))
        
        X = X.add(self.hidden2Activation(self.hidden2(X)))
        
        X = torch.sigmoid(self.hidden3(X))
        
        return X

class ToxicClassifierFitter():
    def __init__(self,
                 optimizer,
                 error,
                 model,
                 vectors,
                 device,
                 EPOCHS = 2,
                 seed_acc = 0.9,
                 save_checkpoint = False,
                 model_save_location = "TCM_3.pt"
                 ):
        self.optimizer = optimizer
        self.error = error
        self.model = model
        self.EPOCHS = EPOCHS
        self.acc = seed_acc
        self.vectors = vectors
        self.device = device
        self.model_save_location = model_save_location
        self.save_checkpoint = save_checkpoint
    
    def accuracy(self, batched_X_test, batched_y_test):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(zip(batched_X_test, batched_y_test)):
            var_X_batch = Variable(torch.nn.utils.rnn.pad_sequence([ self.vectors[X] for X in X_batch]).permute(1,0,2)).float().to(self.device)
            var_y_batch = Variable(torch.from_numpy(y_batch)).float().to(self.device)
            output = self.model(var_X_batch)

            # Total correct predictions
            predicted = output.data.round()
            correct += (predicted == var_y_batch).sum()
            del var_X_batch
            del var_y_batch
            del output
            del predicted
            torch.cuda.empty_cache()
            
        self.acc = float(correct*100) / float(6 * BATCH_SIZE * len(batched_X_test))
            
        return self.acc

    def F1Score(self, batched_X_test, batched_y_test):
        preds = []
        truePreds = []
        for batch_idx, (X_batch, y_batch) in enumerate(zip(batched_X_test, batched_y_test)):
            var_X_batch = Variable(torch.nn.utils.rnn.pad_sequence([ vectors[X] for X in X_batch]).permute(1,0,2)).float().to(device)
            var_y_batch = Variable(torch.from_numpy(y_batch)).float().to(device)
            output = self.model(var_X_batch)

            preds = preds + [ round(float(x)) for X in output.data for x in X ]
            truePreds = truePreds + [ round(float(x)) for X in var_y_batch for x in X ]
            
        return f1_score(truePreds, preds)

    def predict(self, batched_X_test, batched_y_test):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(zip(batched_X_test, batched_y_test)):
            var_X_batch = Variable(torch.nn.utils.rnn.pad_sequence([ self.vectors[X] for X in X_batch]).permute(1,0,2)).float().to(self.device)
            var_y_batch = Variable(torch.from_numpy(y_batch)).float().to(self.device)
            output = self.model(var_X_batch)

            # Total correct predictions
            predicted = output.data.round()
            del var_X_batch
            del var_y_batch
            del output
            del predicted
            torch.cuda.empty_cache()
            
        return predicted
    
    def fit(self, batched_X_train, batched_y_train):
        for epoch in range(self.EPOCHS):
            correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(zip(batched_X_train, batched_y_train)):
                var_X_batch = Variable(torch.nn.utils.rnn.pad_sequence([ self.vectors[X] for X in X_batch]).permute(1,0,2)).float().to(self.device)
                var_y_batch = Variable(torch.from_numpy(y_batch)).float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(var_X_batch)
                loss = self.error(output, var_y_batch)
                loss.backward()
                self.optimizer.step()

                # Total correct predictions
                predicted = output.data.round()
                correct += (predicted == var_y_batch).sum()
                #print(correct)
                if batch_idx % 50 == 0:
                    nni.report_intermediate_result(float(correct*100) / float(6 * BATCH_SIZE*(batch_idx+1)))
                    #print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    #    epoch, batch_idx*len(X_batch), len(batched_X_train), 100.*batch_idx / len(batched_X_train), loss.data, float(correct*100) / float(6 * BATCH_SIZE*(batch_idx+1))))
                del var_X_batch
                del var_y_batch
                del loss
                del output
                del predicted
                torch.cuda.empty_cache()

    def fitF1(self, batched_X_train, batched_y_train):
        for epoch in range(self.EPOCHS):
            preds = []
            truePreds = []
            for batch_idx, (X_batch, y_batch) in enumerate(zip(batched_X_train, batched_y_train)):
                var_X_batch = Variable(torch.nn.utils.rnn.pad_sequence([ self.vectors[X] for X in X_batch]).permute(1,0,2)).float().to(self.device)
                var_y_batch = Variable(torch.from_numpy(y_batch)).float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(var_X_batch)
                loss = self.error(output, var_y_batch)
                loss.backward()
                self.optimizer.step()

                preds = preds + [ round(float(x)) for X in output.data for x in X ]
                truePreds = truePreds + [ round(float(x)) for X in var_y_batch for x in X ]

                if batch_idx % 50 == 0:
                    nni.report_intermediate_result(f1_score(truePreds, preds))
                del var_X_batch
                del var_y_batch
                del loss
                del output
                torch.cuda.empty_cache()

def createFitter(LSTM_UNITS = 128,
                 #hidden1Size = 512,
                 #hidden2Size = 512,
                 dropout_rate = 0.2,
                 hidden1Activation = F.relu,
                 hidden2Activation = F.relu,
                 learning_rate=0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 amsgrad=False,
                 weight_decay=0,
                 epochs = 1,
                 model_save_location="TCM_3.pt",
                 vectors=vectors
                 ):
    
    # get device
    
    model = ToxicClassifierModel(LSTM_UNITS = LSTM_UNITS,
                                 dropout_rate = dropout_rate,
                                 hidden1Activation = hidden1Activation,
                                 hidden2Activation = hidden2Activation,
                                 #hidden1Size = hidden1Size,
                                 #hidden2Size = hidden2Size
                                )
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1,beta_2), amsgrad=amsgrad, weight_decay=weight_decay)
    
    error = nn.BCELoss()
    
    # return final fitter 

    return ToxicClassifierFitter(optimizer, error,
                                 model,
                                 vectors,
                                 device,
                                 EPOCHS = epochs,
                                 model_save_location = model_save_location,
                                 save_checkpoint = True)

def get_default_parameters():
    '''get default parameters'''
    params = {'LSTM_UNITS' : 128
             #,'hidden1Size' : 460
             #,'hidden2Size' : 175
             ,'dropout_rate' : 0.6874797143307255
             ,'hidden1Activation' : "relu"
             ,'hidden2Activation' : "selu"
             ,'learning_rate' : 0.007588713323581429
             ,'amsgrad' : False
             ,'weight_decay': 0.017485451115946377
             ,'epochs': 1
             }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    activation_dict = {'relu':F.relu
                      ,'leaky_relu':F.leaky_relu
                      ,'softmax':F.softmax
                      ,'selu':F.selu
                      ,'tanh':F.tanh
                      ,'sigmoid':F.sigmoid
                      #,'linear':F.linear
                      ,'elu':F.elu
                      }

    model = createFitter(
                 LSTM_UNITS = 60, # PARAMS.get('LSTM_UNITS'),
                 #hidden1Size = PARAMS.get('hidden1Size'),
                 #hidden2Size = PARAMS.get('hidden2Size'),
                 dropout_rate = PARAMS.get('dropout_rate'),
                 hidden1Activation = activation_dict[PARAMS.get('hidden1Activation')],
                 hidden2Activation = activation_dict[PARAMS.get('hidden2Activation')],
                 learning_rate = PARAMS.get('learning_rate'),
                 amsgrad = PARAMS.get('amsgrad'),
                 weight_decay = PARAMS.get('weight_decay') if PARAMS.get('weight_Q') else 0,
                 epochs = 1 # PARAMS.get('epochs')
                 )

    return model

def run(X_train, y_train, X_test, y_test, model):
    '''Train model and predict result'''
    model.fitF1(X_train, y_train) # model.fit(X_train, y_train)
    score = model.F1Score(X_test, y_test) # model.accuracy(X_test, y_test)
    LOG.debug('Accuracy score: %s' % score)
    nni.report_final_result(score)

if __name__ == '__main__':
    batched_X_train, batched_y_train, batched_X_test, batched_y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(batched_X_train, batched_y_train, batched_X_test, batched_y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
