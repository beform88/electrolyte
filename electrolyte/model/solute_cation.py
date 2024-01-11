import numpy as np
import torch
import torch.nn as nn

'''
Input:
    关于金属阳离子的描述符(belike: 电子轨道信息， 电子数， 电子位置， 
                        电子电荷， 电子电荷密度， 电子电子相互作用， 电子电子相互作用密度，
                        电子电子相互作用电荷， 电子电子相互作用电荷密度， 电子电子相互作用电荷密度， 电子电子相互作用电荷)
    - in: torch.Tensor, shape (batch_size, input_dim)

Output:
    Embedding 向量
    - out: torch.Tensor, shape (batch_size, embedding_dim)
'''

class SoluteCation_EmbedModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, embedding_layer_num = 1):
        super(SoluteCation_EmbedModel, self).__init__()
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