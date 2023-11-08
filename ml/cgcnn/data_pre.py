import pandas as pd
import numpy as np

data = pd.read_csv("COF_origin_DC.csv")

names = data["structure"]
# DC_compare = []
for name in names:
    ori = data[data["structure"]==name][["111K","231K","296K"]]
    one_h_data = pd.read_csv("COF_one_0.5_DC.csv")
    one_f_data = pd.read_csv("COF_one_1.0_DC.csv")
    two_h_data = pd.read_csv("COF_two_0.5_DC.csv")
    two_f_data = pd.read_csv("COF_two_1.0_DC.csv")

    one_h = one_h_data[one_h_data["structure"]==name][["111K","231K","296K"]]
    if one_h.empty:
        one_h = [0,0,0]
    one_f = one_f_data[one_f_data["structure"]==name][["111K","231K","296K"]]
    if one_f.empty:
        one_f = [0,0,0]
    two_h = two_h_data[two_h_data["structure"]==name][["111K","231K","296K"]]
    if two_h.empty:
        two_h = [0,0,0]
    two_f = two_f_data[two_f_data["structure"]==name][["111K","231K","296K"]]
    if two_f.empty:
        two_f = [0,0,0]

    ori = np.array(ori).flatten()
    one_h = np.array(one_h).flatten()
    one_f = np.array(one_f).flatten()
    two_h = np.array(two_h).flatten()
    two_f = np.array(two_f).flatten()

    all_DC = np.hstack((ori,one_h,one_f,two_h,two_f))
    # DC_compare.append(all_DC)
    
    np.save("./data/json/npy/" + name + ".npy", all_DC)

# df_DC_compare = pd.DataFrame(DC_compare)
# df_DC_compare.to_csv("DC_all.csv")