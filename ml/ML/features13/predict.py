import pandas as pd
import numpy as np
import joblib

def prediction(fea_file = "all_features_pre.csv",
               model_names = ["origin","one_0.5","one_1.0","two_0.5","two_1.0"],
               T = [1,2,3],
               pre_file_name = "predict.xlsx"):
    features_all = pd.read_csv(fea_file,usecols = ["Name","PLD","LCD","VSA","Density","VF",
                                                   "H","C","N","O","Metal","N_Metal",
                                                   "N_group"])

    pre_file = pd.ExcelWriter(pre_file_name)
    
    for model_name in model_names:
        
        print("model name: ", model_name)
        
        if model_name =="origin":
            
            model_name = joblib.load(model_name + ".pkl")

            pre_all_t = []
            for t in T:
                fea = features_all.iloc[1:,1:]
                fea["T"] = t
                pre = model_name.predict(fea)
                pre_all_t.append(pre)

            df_pre = pd.DataFrame({"Name":features_all.iloc[1:,0:1].values.flatten(),"pre_111":pre_all_t[0],
                                   "pre_231":pre_all_t[1],"pre_296":pre_all_t[2]})
            df_pre.to_excel(pre_file, index=False, sheet_name = "origin")
        else:

            name = model_name
            
            model_name = joblib.load(model_name + ".pkl")
            
            features_new = features_all[features_all["N_group"]>0]

            pre_all_t = []
            for t in T:
                fea = features_new.iloc[1:,1:]
                fea["T"] = t
                pre = model_name.predict(fea)
                pre_all_t.append(pre)
                
            df_pre = pd.DataFrame({"Name":features_new.iloc[1:,0:1].values.flatten(),"pre_111":pre_all_t[0],
                                   "pre_231":pre_all_t[1],"pre_296":pre_all_t[2]})
            df_pre.to_excel(pre_file, index=False, sheet_name = name)

    pre_file.close()
