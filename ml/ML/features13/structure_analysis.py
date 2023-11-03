from mofun import Atoms, find_pattern_in_structure
import pandas as pd
import os
from tqdm import tqdm

def find_group(cifs_folder = "./structure/test/",group = "./structure/group/query.cml", saveto: str = "structure_features.csv")->pd.DataFrame:

    files_name = os.listdir(cifs_folder)
    cifs_name = []
    for cif_name in files_name:
        if os.path.splitext(cif_name)[1] == '.cif':
            cifs_name.append(os.path.splitext(cif_name)[0])
        else:
            pass
        
    all_number = []
    cluster = Atoms.load(group)
    print("cluster group in: " + group)
    for structure_name in tqdm(cifs_name):
        
        structure = Atoms.load(cifs_folder + structure_name + ".cif")
        print("start: " + structure_name + "\n", end = "", flush = True)
        print("get structure: " + cifs_folder + structure_name + ".cif", end = "", flush = True)
        
        result = find_pattern_in_structure(structure, cluster, return_positions_and_quats=False, atol=1)
        all_number.append([structure_name, len(result)])

    df_all_number = pd.DataFrame(all_number, columns=["Name",'N_group'])
    
    if saveto:
        df_all_number.to_csv(saveto, index=True, index_label='Number')
        
    return df_all_number
