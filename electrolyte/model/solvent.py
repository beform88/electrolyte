import numpy as np
import torch
import torch.nn as nn

'''
Input:
    关于阴离子的描述符
    - in: torch.Tensor, shape (batch_size, input_dim)

Output:
    Embedding 向量
    - out: torch.Tensor, shape (batch_size, embedding_dim)
'''

class Solvent_EmbedModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, embedding_layer_num = 1):
        '''
        Args:
            input_dim: int, 输入维度
            embedding_dim: int, 输出维度
            embedding_layer_num: int, 隐藏层数
        '''
        super(Solvent_EmbedModel, self).__init__()
        self.begin = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )
        self.embedding = nn.Sequential()
        for i in range (embedding_layer_num):
            self.embedding.append(nn.Linear(embedding_dim, embedding_dim))
            self.embedding.append(nn.ReLU())

    def forward(self, x):
        x = self.begin(x)
        return self.embedding(x)