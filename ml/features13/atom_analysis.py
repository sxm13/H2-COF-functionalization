import mofun
from mofun import Atoms, find_pattern_in_structure
import os
import argparse
import pandas as pd
import tqdm
import numpy as np
import re

"""
    number of hydrogen atoms per unit cell
    number of carbon atoms per unit cell
    number of nitrogen atoms per unit cell
    number of oxygen atoms per unit cell
    metal type
    metalic_perct
"""

def atom_dict(cif_file):
    
    atoms = []
    count = 0
    with open(cif_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len((re.split(r"[ ]+", line))) == 5:
                atoms.append(re.split(r"[ ]+", line)[0])
            else:
                pass
    f.close()
    return atoms

def metal_count(atoms):

    metals = ['Co', 'Cu', 'Ni', 'V', 'Zn']

    metal_count = 0
    metal_all = []
    for atom_type in atoms:
        if atom_type in metals:
            metal_all.append(atom_type)
            metal_count += 1
    if metal_count > 0:
        name = metal_all[0]
    else:
        name = "None"
            
    return name, metal_count

def gather_features_main(cifs_folder = "./structure/test/", saveto: str = "atom_features.csv")->pd.DataFrame:

    element_feats = ['H', 'C', 'N', 'O']
    metal_types = ['Co', 'Cu', 'Ni', 'V', 'Zn']
    
    files_name = os.listdir(cifs_folder)
    cifs_name = []
    for cif_name in files_name:
        if os.path.splitext(cif_name)[1] == '.cif':
            cifs_name.append(os.path.splitext(cif_name)[0])
        else:
            pass
    all_data = []
    for structure in cifs_name:
        atoms = atom_dict(cifs_folder + structure + ".cif")
        
        metal, number = metal_count(atoms)
        
        N_ele = np.zeros(len(element_feats))
        for i,element in enumerate(element_feats):
            for atom in atoms:
                if atom == element:
                    N_ele[i] += 1

        data = [structure, N_ele[0], N_ele[1], N_ele[2], N_ele[3],metal, number]
        all_data.append(data)
    df_data = pd.DataFrame(all_data, columns=["Name",'H','C','N','O', 'Metal', 'N_Metal'])
    
    if saveto:
        df_data.to_csv(saveto, index=True, index_label='Number')

    return df_data
