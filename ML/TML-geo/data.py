import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def wt(a):
    b = 100*(0.001*a)/(1+0.001*a)
    return b

def gL(a):
    b = a/11.2
    return b

T_list = [111,231,296]
def all_pre(data_csv,fea_csv,unit):
    ori_data = pd.read_csv(data_csv)
    ori_fea = pd.read_csv(fea_csv)
    dataset = []
    for name in ori_data["name"]:
        fea = ori_fea[ori_fea["name"]==name][["name","PLD","LCD","ASA","Density","VF"]].values.tolist()
        for t in T_list:
            diff_t = []
            if unit =="wt":
                tar = wt(ori_data[ori_data["name"]==name][str(t)+"_"+str("1e7")]-ori_data[ori_data["name"]==name][str(t)+"_"+str("5e5")])
                tar = float(tar.iloc[0])
            else:
                tar = gL(ori_data[ori_data["name"]==name][str(t)+"_"+str("1e7")]-ori_data[ori_data["name"]==name][str(t)+"_"+str("5e5")])
                tar = float(tar.iloc[0])
            diff_t.append([fea[0][0],fea[0][1],fea[0][2],fea[0][3],fea[0][4],fea[0][5],str(t),str(tar)])
            dataset.append(diff_t[0])
    df_dataset = pd.DataFrame(dataset,columns= ["name","PLD","LCD","ASA","Density","VF","T","DC"])
    print(df_dataset.head(5))
    df_dataset.to_csv("dataset_"+unit+".csv")

def data_split(data_csv):
    data = pd.read_csv(data_csv)
    X = data[["PLD","LCD","ASA","Density","VF","T"]]
    Y = data["DC"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1129)
    return X_train, X_test, y_train, y_test

def normal(data_csv):
    Xtrain, Xtest, Ytrain, Ytest = data_split(data_csv)
    scaler = StandardScaler()
    try:
        scaler = joblib.load("scaler.gz")
        Xtrain = scaler.transform(Xtest)
        Xtest = scaler.transform(Xtest)
    except:
        Xtrain = scaler.fit_transform(Xtrain)
        joblib.dump(scaler, "scaler.gz")
        scaler = joblib.load("scaler.gz")
        Xtest = scaler.transform(Xtest)
    df_Xtrain = pd.DataFrame(Xtrain,columns=["PLD","LCD","AVSA","Density","VF","T"])
    df_Ytrain = pd.DataFrame(Ytrain,columns=["DC"])
    df_Xtest = pd.DataFrame(Xtest,columns=["PLD","LCD","AVSA","Density","VF","T"])
    df_Ytest = pd.DataFrame(Ytest,columns=["DC"])
    return df_Xtrain, df_Xtest, df_Ytrain, df_Ytest
