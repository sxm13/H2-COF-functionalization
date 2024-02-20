import numpy as np
import pandas as pd

def wt(a):
    b = 100*(0.001*a)/(1+0.001*a)
    return b

def gL(a):
    b = a/11.2
    return b

T_list = [111,231,296]
def all_pre(data_csv,unit):
    data = pd.read_csv(data_csv)
    for name in data["name"]:
        if unit =="wt":
            data_list = []
            for t in T_list:
                tar = wt(data[data["name"]==name][str(t)+"_"+str("1e7")]-data[data["name"]==name][str(t)+"_"+str("5e5")])
                tar = float(tar.iloc[0])
                data_list.append(tar)
            DC = np.array(data_list).flatten()
            np.save("./data/cif/npy" + "_" + unit + "/" + name + ".npy", DC)
        else:
            data_list = []
            for t in T_list:
                tar = gL(data[data["name"]==name][str(t)+"_"+str("1e7")]-data[data["name"]==name][str(t)+"_"+str("5e5")])
                tar = float(tar.iloc[0])
                data_list.append(tar)
            DC = np.array(data_list).flatten()
            np.save("./data/cif/npy" + "_" + unit + "/" + name + ".npy", DC)
            
# all_pre(data_csv="uptake_w.csv",unit="wt")