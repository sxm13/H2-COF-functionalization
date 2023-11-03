import numpy as np
import pandas as pd

data = pd.read_csv("COF_origin_DC.csv")

names = data["structure"]

for name in names:
    a = np.load("./data/json/npy/" + name + ".npy")
    if a.shape[0]!=15:
        print(name,a)