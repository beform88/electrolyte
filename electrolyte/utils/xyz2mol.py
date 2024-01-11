from hashlib import md5
import numpy as np
from openbabel import openbabel
from rdkit import Chem
from ase.io import read, write
import os

def xyz2mol(xyz = '',save_path=None, method = 'opb'):
    
    if method == 'opb':
        obConversion = openbabel.OBConversion()

        # Read in XYZ file
        obConversion.SetInFormat("xyz")
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, xyz)

        # Write out MOL file
        obConversion.SetOutFormat("mol")
        if save_path == None:
            save_path = os.path.splitext(xyz)[0]+'.mol'
            obConversion.WriteFile(mol, save_path)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            # mol = read(save_path)
            os.remove(save_path)
        else:
            obConversion.WriteFile(mol, save_path)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            # mol = read(save_path)

    elif method == 'ase':
        molecules = read(xyz)
        if save_path == None:
            save_path = 'temp_mol.mol'
            write(save_path, molecules)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            os.remove(save_path)
        else:
            write(save_path, molecules)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)

    return mol