import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchnlp.word_to_vector import FastText

# use FastText to vectorize text
vectors = FastText()

# parameters for model
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

class ToxicClassifierModel(nn.Module):
    """[Base class for all neural network modules.]

    Args:
        nn ([object]): [Neural network model using PyTorch NLP.
        Modules can also contain other Modules, allowing to nest them in a tree structure.]
    """
    def __init__(self):
        super(ToxicClassifierModel, self).__init__()
        self.BiGRU = nn.GRU(300, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=1)
        self.BiRNN = nn.RNN(input_size = 2 * LSTM_UNITS, hidden_size = LSTM_UNITS, bidirectional=True)
        self.hidden1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden3 = nn.Linear(DENSE_HIDDEN_UNITS, 6)

    def forward(self, x):
        """[Feed forward network. It takes the input and feeds it through several layers
        one after the other, and returns an output]

        Args:
            X ([torch.float32]): [A vector matrix]

        Returns:
            [Object]: [description]
        """
        x = x.permute(0, 2, 1)
        x = F.dropout2d(X, 0.2, training=self.training)
        x = x.permute(0, 2, 1)
        x = self.BiGRU(X)
        x = self.BiRNN(X[0])
        x = x[0]
        x = torch.cat((torch.max(x, 1).values, torch.mean(x, 1)), 1)
        x = x.add(F.relu(self.hidden1(x)))
        x = x.add(F.relu(self.hidden2(x)))
        x = torch.sigmoid(self.hidden3(x))
        return x