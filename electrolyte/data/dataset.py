# 定义数据集
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
import torch
import random
from desc import *

class ElectrolyteDataset(Dataset):
    def __init__(self, data_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.finetune_unimol_model = False
        
        csv_data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.search_space = csv_data.columns
        self.data = csv_data.to_numpy()
        self.property = self.scaler.fit_transform(self.data[:,-4:])
        self.cation_conc = self.data[:,:1]
        self.anion_conc = self.data[:,1:2]
        self.solvent_conc = self.data[:,2:-4]

        self.cation_types = self.search_space[:1]
        self.anion_types = self.search_space[1:2]
        self.solvent_types = self.search_space[2:-4]

        f_read = open('/vepfs/fs_users/ycjin/electrolyte/data/structure/solvent_input_dict.pkl', 'rb')
        self.solvent_structure = pickle.load(f_read)
        f_read.close()

        f_read = open('/vepfs/fs_users/ycjin/electrolyte/data/structure/anion_input_dict.pkl', 'rb')
        self.anion_structure = pickle.load(f_read)
        f_read.close()

        self.unirepr_Gen = UniRepr_Generator(finetune = True)

        if not self.finetune_unimol_model:
            self.solvent_descs = self.trans_Solvent2Desc().detach()
            self.anion_descs = self.trans_Anion2Desc().detach()
            self.cation_descs = self.trans_Cation2Desc()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        property = torch.tensor(self.property[index]).float()
        
        solvent_conc = torch.tensor(self.solvent_conc[index])
        anion_conc = torch.tensor(self.anion_conc[index])
        cation_conc = torch.tensor(self.cation_conc[index])

        if self.finetune_unimol_model:
            solvent_descs = self.trans_Solvent2Desc()
            anion_descs = self.trans_Anion2Desc()
            cation_descs = self.trans_Cation2Desc()
        else:
            solvent_descs = self.solvent_descs.float()
            anion_descs = self.anion_descs.float()
            cation_descs = self.cation_descs.float()

        solvent_remain_idx = self.gen_remain_index(solvent_conc)
        solvent_conc = torch.index_select(solvent_conc, 0, solvent_remain_idx).float()
        solvent_descs = torch.index_select(solvent_descs, 0, solvent_remain_idx.to(self.device)).float()

        solvent_shuffle_idx = self.shuffle_idx(solvent_conc.shape[0])
        solvent_conc = torch.index_select(solvent_conc, 0, solvent_shuffle_idx).float()
        solvent_descs = torch.index_select(solvent_descs, 0, solvent_shuffle_idx.to(self.device)).float()

        anion_shuffle_idx = self.shuffle_idx(anion_conc.shape[0])
        anion_conc = torch.index_select(anion_conc, 0, anion_shuffle_idx).float()
        anion_descs = torch.index_select(anion_descs, 0, anion_shuffle_idx.to(self.device)).float()

        cation_shuffle_idx = self.shuffle_idx(cation_conc.shape[0])
        cation_conc = torch.index_select(cation_conc, 0, cation_shuffle_idx).float()
        cation_descs = torch.index_select(cation_descs, 0, cation_shuffle_idx).float()

        conc = torch.concat([solvent_conc,anion_conc,cation_conc]).float()

        conc_cilp = torch.tensor([solvent_conc.shape[0], anion_conc.shape[0], cation_conc.shape[0]]).int()

        return property.to(self.device), solvent_descs, anion_descs, cation_descs.to(self.device), conc.to(self.device), conc_cilp
        # return property.to(self.device), torch.rand_like(solvent_descs), torch.rand_like(anion_descs), torch.rand_like(cation_descs).to(self.device), conc.to(self.device), conc_cilp  
    
    def trans_Solvent2Desc(self):

        desc = []
        for k in self.solvent_structure.keys():
            input = self.solvent_structure[k]
            desc.append(self.unirepr_Gen.get2fintune(input,'molecule',False))

        return torch.stack(desc)

    def trans_Anion2Desc(self):

        desc = []
        for k in self.anion_structure.keys():
            input = self.anion_structure[k]
            desc.append(self.unirepr_Gen.get2fintune(input,'molecule',False))

        return torch.stack(desc)

    def trans_Cation2Desc(self):
        desc = []
        for k in self.cation_types:
            if k == 'Li':desc.append(torch.tensor([2,0,0,0,0,0,0]))
        return torch.stack(desc)
    
    def gen_remain_index(self,conc,remain = 4):
        zero_inedx = torch.where(conc == 0)[0]
        no_zero_inedx = torch.where(conc!= 0)[0]
        zero_inedx_remain = torch.Tensor(random.sample(zero_inedx.tolist(), remain - no_zero_inedx.shape[0] )).int()
        return torch.cat([no_zero_inedx, zero_inedx_remain])
    
    def shuffle_idx(self,lens):
        return torch.randperm(lens)



dataset = ElectrolyteDataset('/vepfs/fs_users/ycjin/electrolyte/data/raw_data/data_percent.csv')
property, solvent_descs, anion_descs, cation_descs, conc, conc_clip = dataset[0]