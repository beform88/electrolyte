import sys
import numpy as np 
sys.path.append('/vepfs/fs_users/ycjin/Delta-ML-Framework/Unimol_2_NMR_fix/descriptior/unimol_tools')
from unimol_tools import UniMolRepr, UniMolRepr_F
from unimol_tools.data import Coords2Unimol
from ase.io import read
from rdkit import Chem
import torch

class UniRepr_Generator(object):
    def __init__(self,base_type = 'mol', finetune = False):
        if finetune:
            self.generator = UniMolRepr_F(data_type='molecule', remove_hs=False, base_type = base_type, no_optimize = True)
        else:
            self.generator = UniMolRepr_F(data_type='molecule', remove_hs=False, base_type = base_type, no_optimize = True)
        
        # self.coords2unimol = Coords2Unimol()

    def UniRepr_atom(self, mol, atom_id, molecule_repr = True):
        if len(mol) > 1:
            uni_desc = self.generator.get_repr(mol)
            uni_descs = []
            if molecule_repr:
                for i in range(len(uni_desc['cls_repr'])):
                    uni_descs.append(uni_desc['cls_repr'][i] + uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
            else:
                for i in range(len(uni_desc['cls_repr'])):
                    uni_descs.append(uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
            return uni_descs
        else: 
            uni_desc = self.generator.get_repr([mol])
            return uni_desc['cls_repr'][0] + uni_desc['atomic_reprs'][0][atom_id[0]-1].tolist()

    def UniRepr_molecule(self, mol, atom_repr = False):
        # if len(mol) > 1:
        #     uni_desc = self.generator.get_repr(mol)
        # else:
        uni_desc = self.generator.get_repr(mol)

        if atom_repr == False:
            return uni_desc['cls_repr']
        else:
            return uni_desc['cls_repr'], uni_desc['atomic_reprs']
        
    def get2train(self, input, desc_level, only_atom_repr = True):
        max_num = 500
        atom_id = input['atom']
        uni_descs = []
        if input['unimol']['src_tokens'].shape[0] < max_num:
            uni_desc = self.generator.get_reprs(input['unimol'])
            for i in range(len(uni_desc['cls_repr'])):
                if desc_level == 'atom':
                    if only_atom_repr == True:
                        uni_descs.append(uni_desc['cls_repr'][i].tolist() + uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
                        # return torch.cat([uni_desc['cls_repr'],uni_desc['atomic_reprs'][atom_id-1]],axis = 0)
                    else:
                        uni_descs.append(uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
                        # return uni_desc['atomic_reprs'][0][atom_id-1]
                elif desc_level == 'molecule':
                    uni_descs.append(uni_desc['cls_repr'][i].tolist())
                    # return uni_desc['cls_repr'][0]
                else:
                    raise ValueError('UnKnown Desc Level, u should use atom or molecule')
        else:
            for i in range(input['unimol']['src_tokens'].shape[0] // max_num + 1):
                batch = {}
                for k in input['unimol'].keys():
                    batch[k] = input['unimol'][k][i*max_num:(i+1)*max_num]
                uni_desc = self.generator.get_reprs(batch)

                for i in range(len(uni_desc['cls_repr'])):
                    if desc_level == 'atom':
                        if only_atom_repr == True:
                            uni_descs.append(uni_desc['cls_repr'][i].tolist() + uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
                            # return torch.cat([uni_desc['cls_repr'],uni_desc['atomic_reprs'][atom_id-1]],axis = 0)
                        else:
                            uni_descs.append(uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
                            # return uni_desc['atomic_reprs'][0][atom_id-1]
                    elif desc_level == 'molecule':
                        uni_descs.append(uni_desc['cls_repr'][i].tolist())
                        # return uni_desc['cls_repr'][0]
                    else:
                        raise ValueError('UnKnown Desc Level, u should use atom or molecule')

        return uni_descs
        
    def get2fintune(self, input, desc_level, atom_repr = True):
        uni_desc = self.generator.get_repr(input)
        if desc_level == 'atom':
            atom_id = input['atom']
            if atom_repr == True:
                return torch.cat([uni_desc['cls_repr'][0],uni_desc['atomic_reprs'][0][atom_id-1]],axis = 0)
            else:
                return uni_desc['atomic_reprs'][0][atom_id-1]
        elif desc_level == 'molecule':
            return uni_desc['cls_repr'][0]
        else:
            raise ValueError('UnKnown Desc Level, u should use atom or molecule')
        
    def get_models(self):
        return 'unimol', self.generator.model

    def mols2src(self):
        pass
