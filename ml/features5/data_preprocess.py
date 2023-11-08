import pandas as pd
import numpy as np

def wt(a):
    b = 100*(0.001*a)/(1+0.001*a)
    return b

def g_L(a):
    b = a/11.2
    return b

def process(path = "./Result/", target = "wt", cif_stru = "origin", ratio = "0"):

    if target == "wt":
        
        if cif_stru == "origin":
            data = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "g")
            data_name = data["structure"]
            data_111 = wt(data["111_1e7"] - data["111_5e5"])
            data_231 = wt(data["231_1e7"] - data["231_5e5"])
            data_296 = wt(data["296_1e7"] - data["296_5e5"])

            data_all = [data_name, data_111, data_231, data_296]
            df_data_all = pd.DataFrame({'structure': data_all[0],
                                       '111K': data_all[1],
                                       '231K': data_all[2],
                                       '296K': data_all[3]})
            df_data_all.to_csv("./ML/features5/" + "COF_" + cif_stru + "_DC_" + target + ".csv",index=True, index_label='Number')

        else:
            data_1 = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "g_0.5")
            data_2 = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "g_1.0")

            data_name_1 = data_1["structure"]
            data_111_0 = wt(data_1["111_1e7"] - data_1["111_5e5"])
            data_231_0 = wt(data_1["231_1e7"] - data_1["231_5e5"])
            data_296_0 = wt(data_1["296_1e7"] - data_1["296_5e5"])

            data_name_2 = data_2["structure"]
            data_111_1 = wt(data_2["111_1e7"] - data_2["111_5e5"])
            data_231_1 = wt(data_2["231_1e7"] - data_2["231_5e5"])
            data_296_1 = wt(data_2["296_1e7"] - data_2["296_5e5"])

            data_all_1 = [data_name_1, data_111_0, data_231_0, data_296_0]
            data_all_2 = [data_name_2, data_111_1, data_231_1, data_296_1]

            df_data_all_1 = pd.DataFrame({'structure': data_all_1[0],
                                       '111K': data_all_1[1],
                                       '231K': data_all_1[2],
                                       '296K': data_all_1[3]})
            df_data_all_2 = pd.DataFrame({'structure': data_all_2[0],
                                       '111K': data_all_2[1],
                                       '231K': data_all_2[2],
                                       '296K': data_all_2[3]})

            df_data_all_1.to_csv("./ML/features5/" + "COF_" + cif_stru + "_0.5_DC_" + target + ".csv",index=True, index_label='Number')
            df_data_all_2.to_csv("./ML/features5/" + "COF_" + cif_stru + "_1.0_DC_" + target + ".csv",index=True, index_label='Number')
    else:

        if cif_stru == "origin":
            data = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "v")
            data_name = data["structure"]
            data_111 = g_L(data["111_1e7"] - data["111_5e5"])
            data_231 = g_L(data["231_1e7"] - data["231_5e5"])
            data_296 = g_L(data["296_1e7"] - data["296_5e5"])

            data_all = [data_name, data_111, data_231, data_296]
            df_data_all = pd.DataFrame({'structure': data_all[0],
                                       '111K': data_all[1],
                                       '231K': data_all[2],
                                       '296K': data_all[3]})
            df_data_all.to_csv("./ML/features5/" + "COF_" + cif_stru + "_DC_" + target + ".csv",index=True, index_label='Number')

        else:
            data_1 = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "v_0.5")
            data_2 = pd.read_excel(path + cif_stru + "/" + "COF_" + cif_stru +".xlsx", sheet_name = "v_1.0")

            data_name_1 = data_1["structure"]
            data_111_0 = g_L(data_1["111_1e7"] - data_1["111_5e5"])
            data_231_0 = g_L(data_1["231_1e7"] - data_1["231_5e5"])
            data_296_0 = g_L(data_1["296_1e7"] - data_1["296_5e5"])

            data_name_2 = data_2["structure"]
            data_111_1 = g_L(data_2["111_1e7"] - data_2["111_5e5"])
            data_231_1 = g_L(data_2["231_1e7"] - data_2["231_5e5"])
            data_296_1 = g_L(data_2["296_1e7"] - data_2["296_5e5"])

            data_all_1 = [data_name_1, data_111_0, data_231_0, data_296_0]
            data_all_2 = [data_name_2, data_111_1, data_231_1, data_296_1]

            df_data_all_1 = pd.DataFrame({'structure': data_all_1[0],
                                       '111K': data_all_1[1],
                                       '231K': data_all_1[2],
                                       '296K': data_all_1[3]})
            df_data_all_2 = pd.DataFrame({'structure': data_all_2[0],
                                       '111K': data_all_2[1],
                                       '231K': data_all_2[2],
                                       '296K': data_all_2[3]})

            df_data_all_1.to_csv("./ML/features5/" + "COF_" + cif_stru + "_0.5_DC_" + target + ".csv",index=True, index_label='Number')
            df_data_all_2.to_csv("./ML/features5/" + "COF_" + cif_stru + "_1.0_DC_" + target + ".csv",index=True, index_label='Number')

def add_T(path = "./ML/features5/", fea_name = "geometric_features.csv", cif_stru = "origin",target = "wt"):

    fea = pd.read_csv(path + fea_name)
    T_list= ["1","2","3"]
    if cif_stru == "origin":
        data = pd.read_csv(path + "COF_" + cif_stru + "_DC_" + target + ".csv")
        for_ML = []
        for name in data["structure"]:
            try:
                for i in range(3):
                    each_data = []
                    each_data.extend(fea[fea["Name"]==name].iloc[0:1,1:].values[0])
                    each_data.extend(T_list[i])
                    each_data.extend(data[data["structure"]==name].iloc[0:1,2+i:3+i].values[0])
                    for_ML.append(each_data)
            except:
                pass
          
        df_for_ML = pd.DataFrame(for_ML,
                                 columns=["Name","PLD", "LCD", "VSA", "Density", "VF","T","DC"])        
        df_for_ML.to_csv(path + "COF_" + cif_stru + "_ML_" + target + ".csv",index=True, index_label='Number')
    else:
        data1 = pd.read_csv(path + "COF_" + cif_stru + "_0.5_DC_" + target + ".csv")
        data2 = pd.read_csv(path + "COF_" + cif_stru + "_1.0_DC_" + target + ".csv")
        for_ML1 = []
        for_ML2 = []
        for name in data1["structure"]:
            try:
                for i in range(3):
                    each_data1 = []
                    
                    each_data1.extend(fea[fea["Name"]==name].iloc[0:1,1:].values[0])
                    each_data1.extend(T_list[i])
                    each_data1.extend(data1[data1["structure"]==name].iloc[0:1,2+i:3+i].values[0])

                    for_ML1.append(each_data1)
            except:
                pass

        for name in data2["structure"]:
            try:
                for i in range(3):
                    each_data2 = []

                    each_data2.extend(fea[fea["Name"]==name].iloc[0:1,1:].values[0])
                    each_data2.extend(T_list[i])
                    each_data2.extend(data2[data2["structure"]==name].iloc[0:1,2+i:3+i].values[0])
                    
                    for_ML2.append(each_data2)
            except:
                pass
            
        df_for_ML1 = pd.DataFrame(for_ML1,
                                 columns=["Name","PLD", "LCD", "VSA", "Density", "VF", "T","DC"])        
        df_for_ML1.to_csv(path + "COF_" + cif_stru + "_0.5_ML_" + target + ".csv",index=True, index_label='Number')

        df_for_ML2 = pd.DataFrame(for_ML2,
                                 columns=["Name","PLD", "LCD", "VSA", "Density", "VF","T","DC"])        
        df_for_ML2.to_csv(path + "COF_" + cif_stru + "_1.0_ML_" + target + ".csv",index=True, index_label='Number')
    
def merge_all(path = "./ML/features5/", target = "wt",csv_name = "ML_all"):
    
    data_origin = pd.read_csv(path + "COF_origin_ML_" + target + ".csv")
    data_one_0 = pd.read_csv(path + "COF_one_0.5_ML_" + target + ".csv")
    data_one_1 = pd.read_csv(path + "COF_one_1.0_ML_" + target + ".csv")
    data_two_0 = pd.read_csv(path + "COF_two_0.5_ML_" + target + ".csv")
    data_two_1 = pd.read_csv(path + "COF_two_1.0_ML_" + target + ".csv")

    df_origin = data_origin.iloc[0:,1:9]
    df_one_0 = data_one_0.iloc[0:,1:9]
    df_one_1 = data_one_1.iloc[0:,1:9]
    df_two_0 = data_two_0.iloc[0:,1:9]
    df_two_1 = data_two_1.iloc[0:,1:9]

    df_origin["site"] = 0
    df_origin["ratio"] = 0

    df_one_0["site"] = 1
    df_one_0["ratio"] = 0.5

    df_one_1["site"] = 1
    df_one_1["ratio"] = 1

    df_two_0["site"] = 2
    df_two_0["ratio"] = 0.5

    df_two_1["site"] = 2
    df_two_1["ratio"] = 1

    df_all = pd.concat([df_origin, df_one_0, df_one_1, df_two_0, df_two_1], axis=0)

    df_all.to_csv(path + csv_name + "_" + target +".csv", index=True)
