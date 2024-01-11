import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .solvent import Solvent_EmbedModel
from .solute_anion import SoluteAnion_EmbedModel
from .solute_cation import SoluteCation_EmbedModel
from .property import Property_EmbedModel
from .conc_predict_d import ConcPredict_Model

class ElectrolyteModel(nn.Module):
    def __init__(self, **kwargs):
        super(ElectrolyteModel, self).__init__()
        self.property_input_dim = kwargs['property_input_dim']
        self.solute_anion_input_dim = kwargs['solute_anion_input_dim']
        self.solute_cation_input_dim = kwargs['solute_cation_input_dim']
        self.solvent_input_dim = kwargs['solvent_input_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.__init_models_()

    def __init_models_(self, ):
        self.property_embed = Property_EmbedModel(input_dim = self.property_input_dim, embedding_dim = self.hidden_dim)

        self.solvent_embed = Solvent_EmbedModel(input_dim = self.solvent_input_dim + self.hidden_dim, 
                                                embedding_dim = self.hidden_dim)
        
        self.solute_anion_embed = SoluteAnion_EmbedModel(input_dim = self.solute_anion_input_dim + self.hidden_dim, 
                                                         embedding_dim = self.hidden_dim)
        
        self.solute_cation_embed = SoluteCation_EmbedModel(input_dim = self.solute_cation_input_dim + self.hidden_dim, 
                                                           embedding_dim = self.hidden_dim)
        
        self.conc_predict = ConcPredict_Model(input_dim=self.hidden_dim * 7, hidden_dim=self.hidden_dim, output_dim=6)

    def batch_collate_fn(self, batch):
        label = []
        feature = []
        for item in batch:
            label.append(item[-2:])
            feature.append(item[:-2])
        return feature, label

    def forward(self, property, solvent, solute_anion, solute_cation):

        property_embed = self.property_embed(property)

        solvent_out = self.solvent_embed(torch.concat([solvent,property_embed.repeat(solvent.shape[0], 1)], dim = 1))

        solute_anion_out = self.solute_anion_embed(torch.concat([solute_anion,property_embed.repeat(solute_anion.shape[0], 1)], dim = 1))

        solute_cation_out = self.solute_cation_embed(torch.concat([solute_cation,property_embed.repeat(solute_cation.shape[0], 1)], dim = 1))

        self.features = torch.cat([solvent_out, solute_anion_out, solute_cation_out, property_embed.reshape(1,-1)], dim = 0)

        self.features = self.features.reshape([1,-1])

        # self.sub = self.features
        # self.env = self.features.unsqueeze(0).repeat(self.sub.shape[0], 1, 1)

        self.conc = self.conc_predict(self.features)

        return torch.abs(self.conc.reshape([-1]))
# test
    
# init_dict = {
#     'property_input_dim': 256,
#     'solute_anion_input_dim': 256,
#     'solute_cation_input_dim': 256,
#     'solvent_input_dim': 256,
#     'hidden_dim': 256
# }

# property_input = torch.randn(256)
# solvent = torch.randn(2,256)
# solute_anion = torch.randn(3,256)
# solute_cation = torch.randn(4,256)
# model = ElectrolyteModel(**init_dict)
# model(property_input, solvent, solute_anion, solute_cation)