import pandas as pd
import numpy as np

data = pd.read_csv("./data/json-all/COF-all.csv")
names = pd.read_csv("./data/json-all/COF-all.csv")["name"]

for name in names:
    DC = data[data["name"]==name][["111K","231K","296K"]]
    DC = np.array(DC).flatten()

    np.save("./data/json-all/npy/" + name + ".npy", DC)