import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchnlp.word_to_vector import FastText
from torch.nn.utils.rnn import pad_sequence

from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from custom_darts_trainer import DartsTrainer
from nni.nas.pytorch.darts import DartsMutator


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

device = torch.device("cuda:0")

class ToxicClassifierModel(nn.Module):
    def __init__(self):
        super(ToxicClassifierModel, self).__init__()
        self.BiGRU = nn.GRU(300, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=1)
        self.BiRNN = LayerChoice([nn.RNN(input_size = 2 * LSTM_UNITS, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=1),
                                  nn.RNN(input_size = 2 * LSTM_UNITS, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=2)])
        self.hidden1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden3 = nn.Linear(DENSE_HIDDEN_UNITS, 6)
        self.vectors = FastText()
        
        self.skipconnect1 = InputChoice(n_candidates=1)
        self.skipconnect2 = InputChoice(n_candidates=1)
    
    def forward(self, X):
        X = X.permute(0, 2, 1)
        X = F.dropout2d(X, 0.2, training=self.training)
        X = X.permute(0, 2, 1)
        
        X = self.BiGRU(X)
        
        X = self.BiRNN(X[0])
        
        X = X[0]
        
        X = torch.cat((torch.max(X, 1).values, torch.mean(X, 1)), 1)

        X0 = self.skipconnect1([X])
        
        X = F.relu(self.hidden1(X))

        if X0 is not None:
            X += X0

        X0 = self.skipconnect2([X])
        
        X = F.relu(self.hidden2(X))

        if X0 is not None:
            X += X0
        
        X = torch.sigmoid(self.hidden3(X))
        
        return X


def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}

if __name__ == "__main__":
    df = pd.read_csv("toxic-train-kaggle-clean.csv")
    df["word_splits"] = df["word_splits"].apply(eval)
    df = df[(df["word_splits"].apply(len) > 0) & (df["word_splits"].apply(len) <= 560)]

    X_train, X_test, y_train, y_test = train_test_split(df["word_splits"], df.drop("word_splits", axis=1), test_size=0.15)


    net = ToxicClassifierModel()
    print("Toxic Params:", len(list(net.parameters())))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters()) # optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    mutator = DartsMutator(net)
    print("Optimizer defined")

    trainer = DartsTrainer(net,
                           loss=criterion,
                           metrics=accuracy,
                           optimizer=optimizer,
                           num_epochs=2,
                           dataset_train=list(zip(X_train.values, torch.from_numpy(y_train.values))),
                           dataset_valid=list(zip(X_test.values, torch.from_numpy(y_test.values))),
                           batch_size=64,
                           log_frequency=10,
                           mutator=mutator,
                           device=device)
    print("Trainer defined")

    trainer.train()
    trainer.export("checkpoint.json")
