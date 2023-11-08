from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np

def data_prepare(path = "./Result/", cif_stru = "origin", ratio = "0.5"):

    if cif_stru == "origin":
        data = pd.read_csv(path + cif_stru + "/" + "COF_" + cif_stru + "_ML.csv")
    
        X = data.iloc[:,2:15]
        Y = data.iloc[:,15:16]
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    else:
        data = pd.read_csv(path + cif_stru + "/" + "COF_" + cif_stru + "_" + ratio + "_ML.csv")
        
        X = data.iloc[:,2:15]
        Y = data.iloc[:,15:16]
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
