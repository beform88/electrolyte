import numpy as np
import torch
import torch.nn as nn

class ConcPredict_Model(nn.Module):
    def __init__(self,input_dim = None, hidden_dim = None):
        super(ConcPredict_Model, self).__init__()
        # conc model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim * 2, nhead=8), num_layers=6)
        self.conc_head = nn.Sequential()

        for i in range(2):
            self.conc_head.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            self.conc_head.append(nn.ReLU())

        self.conc_head.append(nn.Linear(hidden_dim * 2, 1))

        # env model
        self.env_layer = ConcDeepSet_Model(input_dim = input_dim, output_dim = hidden_dim)

        # sub model
        self.sub_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, sub, env):
        env = self.env_layer(env)
        sub = self.sub_layer(sub)
        x = torch.concat([sub, env], dim = 1)
        x = self.trans_encoder(x)
        x = self.conc_head(x)
        return x
    
class ConcDeepSet_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num = 2):
        super(ConcDeepSet_Model, self).__init__()
        self.begin = nn.Sequential(
            nn.Linear(input_dim+1, output_dim),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential()
        for i in range (hidden_layer_num):
            self.hidden_layer.append(nn.Linear(output_dim, output_dim))
            self.hidden_layer.append(nn.ReLU())

    def forward(self, x):
        x = x.mean(dim = 1)
        x = torch.concat([x.new_ones(x.size(0), 1) * x.shape[0], x], dim = 1)
        x = self.begin(x)
        return self.hidden_layer(x)