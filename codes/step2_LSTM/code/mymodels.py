import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


class MyRNN(nn.Module):
        def __init__(self):
                super(MyRNN, self).__init__()
                # self.rnn = nn.GRU(input_size = 1, hidden_size = 16, num_layers = 1, batch_first = True)
                # self.fc = nn.Linear(in_features = 16, out_features = 5)
                self.rnn = nn.GRU(input_size = 1, hidden_size = 64, num_layers = 5, dropout=0.1, batch_first = True)
                self.fc = nn.Linear(in_features = 64, out_features = 5)
        def forward(self, x):
                #x, _ = self.rnn(x)
                #x = self.fc(x[:, -1, :])
                x, _ = self.rnn(x)
                x = self.fc(F.relu(x[:, -1, :]))
                return x


class MyVariableRNN(nn.Module):
        def __init__(self, dim_input):
                super(MyVariableRNN, self).__init__()
                # You may use the input argument 'dim_input', which is basically the number of features
                self.fc1 = nn.Linear(in_features = dim_input, out_features = 128)
                self.rnn = nn.GRU(input_size=128, hidden_size = 64, num_layers = 3, batch_first = True)
                self.fc2 = nn.Linear(in_features = 64, out_features = 2)

        def forward(self, input_tuple):
                # HINT: Following two methods might be useful
                # 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn

                seqs, lengths = input_tuple
                x = torch.tanh(self.fc1(seqs))
                x = pack_padded_sequence(x, lengths, batch_first = True)
                x, h = self.rnn(x)
                x, _ = pad_packed_sequence(x, batch_first = True)
                #x = x[np.arange(len(x)), lengths-1]
                #x = self.fc2(x)
                x = self.fc2(x[:, -1, :])
                return x
'''
class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        # You may use the input argument 'dim_input', which is basically the number of features
        self.dim_input = dim_input
        self.hidden_dim = 128
        self.bidirectional = True
        self.num_dir = 2 if self.bidirectional else 1
        self.num_layers = 4
        self.dropout = 0.5
        self.out_features = 2

        self.lstm = nn.LSTM(input_size = self.dim_input, hidden_size = self.hidden_dim, num_layers = self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True)
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

        self.gru = nn.GRU(input_size = self.hidden_dim *2, hidden_size = self.hidden_dim, dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
        self.fc = nn.Sequential(
            nn.Linear(1024, int(self.hidden_dim)),
            nn.SELU(True),
            nn.Dropout(p = self.dropout),
            nn.Linear(int(self.hidden_dim), self.out_features),
        )
    def forward(self, input_tuple):
        seqs, lengths = input_tuple
        # print(seqs.shape[2])
        x = seqs
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        avg_pool_l = torch.mean(lstm_out, 1)
        max_pool_l, _ = torch.max(lstm_out, 1)

        avg_pool_g = torch.mean(gru_out, 1)
        max_pool_g, _ = torch.max(gru_out, 1)

        x = torch.cat((avg_pool_g, max_pool_g, avg_pool_l, max_pool_l), 1)
        x = self.fc(x)
        return x
'''
