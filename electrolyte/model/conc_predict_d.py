import numpy as np
import torch
import torch.nn as nn

class ConcPredict_Model(nn.Module):
    def __init__(self,input_dim = None, hidden_dim = None, output_dim = None):
        super(ConcPredict_Model, self).__init__()
        # conc model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.begin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8), num_layers=2)
        self.conc_head = nn.Sequential()

        for i in range(2):
            self.conc_head.append(nn.Linear(hidden_dim, hidden_dim))
            self.conc_head.append(nn.ReLU())

        self.conc_head.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.begin(x)
        x = self.trans_encoder(x)
        x = self.conc_head(x)
        return x