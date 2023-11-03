from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np

def data_prepare(path = "./Result/",filename = "ML_all",target = "wt"):

    data = pd.read_csv(path + filename + "_" + target + ".csv")
    
    X = data[["PLD","LCD","VSA","Density","VF","T","site","ratio"]]
    Y = data["DC"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   

    return X_train, X_test, y_train, y_test
