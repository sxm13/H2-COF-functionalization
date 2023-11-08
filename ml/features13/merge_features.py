import pandas as pd
import numpy as np

def concat_all_features(base_feature = "geometric_features.csv", atom_fea = "atom_features.csv",
                        stru_fea = "structure_features.csv", replace = True, saveto: str = "all_features.csv")->pd.DataFrame:

    geo = pd.read_csv(base_feature)
    atom = pd.read_csv(atom_fea)
    stru = pd.read_csv(stru_fea)

    exact_geo =[]
    exact_atom = []
    exact_stru = []
    for name in geo["Name"]:
        try:
            exact_atom.append(atom[atom["Name"]==name].iloc[0:1,2:].values[0])
            exact_stru.append(stru[stru["Name"]==name].iloc[0:1,2:].values[0])
            exact_geo.append(geo[geo["Name"]==name].iloc[0:1,1:].values[0])
        except:
            pass

    print(exact_geo)
    df_exact_geo = pd.DataFrame(exact_geo,
                                columns=np.delete(geo.columns.values, [0]))
    df_exact_atom = pd.DataFrame(exact_atom,
                                 columns=np.delete(atom.columns.values, [0,1]))
    df_exact_stru = pd.DataFrame(exact_stru,
                                 columns=np.delete(stru.columns.values, [0,1]))

    df_data = pd.concat([df_exact_geo, df_exact_atom, df_exact_stru],axis = 1)
    df_data = df_data.fillna(value=0)

    if replace:
        df_data.replace('Co',int(1),inplace = True)
        df_data.replace('Cu',int(2),inplace = True)
        df_data.replace('Ni',int(3),inplace = True)
        df_data.replace('V',int(4),inplace = True)
        df_data.replace('Zn',int(5),inplace = True)
    if saveto:
        df_data.to_csv(saveto, index=True, index_label='Number')

    return df_data
