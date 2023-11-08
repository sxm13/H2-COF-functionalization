import subprocess
import pandas as pd
import os
from tqdm import tqdm
import re

def zeo_data(cifs_folder = "./structure/test/",zeo_path = '../../../zeo++-0.3/network', verbos=False):

    """Task for obtaining geometric features through using Zeo++0.3 software

        cifs_folder: cifs folder you want to calculated ;
        zeo_path: Zeo++ software installed path and executable file-network :
        verbos: print data or not.
    """
    
    files_name = os.listdir(cifs_folder)
    cifs_name = []
    for cif_name in files_name:
        if os.path.splitext(cif_name)[1] == '.cif':
            cifs_name.append(os.path.splitext(cif_name)[0])
        else:
            pass

    for structure in tqdm(cifs_name):

        cmd_pd = zeo_path + ' -ha -res ' + cifs_folder + structure + ".cif"
        cmd_sa = zeo_path + ' -ha -sa 0 0 5000 ' + cifs_folder + structure + ".cif"
        cmd_pv = zeo_path + ' -ha -volpo 0 0 5000 ' + cifs_folder + structure + ".cif"

        process1 = subprocess.Popen(cmd_pd, stdout=subprocess.PIPE, stderr=None, shell=True)
        process2 = subprocess.Popen(cmd_sa, stdout=subprocess.PIPE, stderr=None, shell=True)
        process3 = subprocess.Popen(cmd_pv, stdout=subprocess.PIPE, stderr=None, shell=True)

        if verbos:
            print("Finish: " + structure)
    
def zeo_result(results_folder = "./structure/test/",remove = True,saveto: str = "geometric_features.csv")->pd.DataFrame:

    """Use for collecting geometric data.
        results_folder: after Zeo++ calculation, the path of result files :
        remove: do you want to remove the Zeo++ result files :
        saveto: csv file for saving data.
    """
    
    files_name = os.listdir(results_folder)
    cifs_name = []
    for cif_name in files_name:
        if os.path.splitext(cif_name)[1] == '.sa':
            cifs_name.append(os.path.splitext(cif_name)[0])
        else:
            pass
    
    all_data = []
    for structure in tqdm(cifs_name):
        
        f1 = open(results_folder + structure + ".res", "r")
        pd_data = f1.readlines()
        for row in pd_data:
            PLD = float(row.split()[2])
            LCD = float(row.split()[1])
        f1.close()

        f2 = open(results_folder + structure + ".sa", "r")
        sa_data = f2.readlines()
        for i,row in enumerate(sa_data):
            if i ==0:
                VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0])
                Density = float(row.split('Density:')[1].split()[0])
        f2.close()

        f3 = open(results_folder + structure + ".volpo", "r")
        pv_data = f3.readlines()
        for i,row in enumerate(pv_data):
            if i ==0:
                VF = float(row.split('POAV_Volume_fraction:')[1].split()[0]) 
        f3.close()
        
        result = [structure, PLD, LCD, VSA, Density, VF]
        all_data.append(result)

        if remove:
            os.remove(results_folder + structure + ".res")
            os.remove(results_folder + structure + ".sa")
            os.remove(results_folder + structure + ".volpo")

    df_data = pd.DataFrame(all_data, columns=["Name",'PLD','LCD','VSA','Density', 'VF'])
    
    if saveto:
        df_data.to_csv(saveto, index=True, index_label='Number')
        
    return df_data
