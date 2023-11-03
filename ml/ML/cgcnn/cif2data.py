import pandas as pd
import pymatgen.core as mg
import numpy as np
import re
import json
import torch
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

def get_structure_data(path,dataset_csv,save_atom_dir):
    periodic_table_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]


    mofs = pd.read_csv(dataset_csv)["name"]
    for mof in mofs:
        try:
            structure = mg.Structure.from_file(path + mof + '.cif')
            elements = [str(site.specie) for site in structure.sites]  

            atom = []
            for i in range(len(elements)):
                ele = elements[i]
                atom_index = periodic_table_symbols.index(ele)
                atom.append(int(int(atom_index)+1))
                np.save(save_atom_dir + mof + '.npy', atom)
        except:
            print("Fail",mof)
        


def CIF2json(path, data_csv, atom_dir,save_path):
       
        mofs = pd.read_csv(data_csv)["name"]

        for mof in mofs:
            try:
                structure = read(path + mof + ".cif")
                struct = AseAtomsAdaptor.get_structure(structure)
                _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
                                                        r=6, numerical_tol=0, exclude_self=True
                                                        )

                _nonmax_idx = []
                for i in range(len(structure)):
                    idx_i = (_c_index == i).nonzero()[0]
                    idx_sorted = np.argsort(n_distance[idx_i])[: 200]
                    _nonmax_idx.append(idx_i[idx_sorted])
                _nonmax_idx = np.concatenate(_nonmax_idx)

                index1 = _c_index[_nonmax_idx]
                index2 = _n_index[_nonmax_idx]
                dij = n_distance[_nonmax_idx]
                numbers = np.load(atom_dir + mof + ".npy")
                nn_num = []

                for i in range(len(structure)):
                    j = 0
                    for idx in range(len(index1)):
                        if index1[idx] == i:
                            j += 1
                        else:
                            pass
                    nn_num.append(j)
            
                data = {"rcut": 6.0,
                        "numbers": numbers.tolist(),
                        "index1": index1.tolist(),
                        "index2":index2.tolist(),
                        "dij": dij.tolist(),
                        "nn_num": nn_num}

                with open(save_path + mof + ".json", 'w') as file:
                    json.dump(data, file)
            except:
                print(mof)

        return mof, index1, index2, dij, _offsets

#get_structure_data(path = "./COF-all/",dataset_csv = "./restart.csv",save_atom_dir = "./atom-all/")
CIF2json(path = "./COF-all/", data_csv = "restart.csv", atom_dir = "./atom-all/",save_path = "./json-all/")