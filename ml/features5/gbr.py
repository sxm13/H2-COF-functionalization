from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import joblib
from ML.features5.data_prepare import data_prepare
import warnings


def training_model(path = "./ML/features5/", file_name = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, cif_stru = cif_stru, target = target)
    
    reg = XGBRegressor()
    space  = [Integer(1,200, name='n_estimators'),
            Integer(1, 10, name='max_depth'),
            Integer(1, 10, name='num_parallel_tree'),
            Integer(1, 10, name='min_child_weight'),
            Real(0.001,1,"log-uniform",name='learning_rate'),
            Real(0.01,1,name='subsample'),
            Real(0.001,10,"log-uniform",name='gamma'),
            Real(0, 1, name='alpha'),
            Real(2, 10, name='reg_alpha'),
            Real(10, 50, name='reg_lambda')
         ]
    @use_named_args(space)

    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
        - n_estimator=%d
        - max_depth=%d
        - num_parallel_tree=%d
        - min_child_weight=%d
        - learning_rate=%f
        - subsample=%f
        - gamma=%f
        - alpha=%f
        - reg_alpha=%f
        - reg_lambda=%f""" % (res_gp.x[0],res_gp.x[1],
                            res_gp.x[2],res_gp.x[3],
                            res_gp.x[4],res_gp.x[5],
                            res_gp.x[6],res_gp.x[7],
                            res_gp.x[8],res_gp.x[9]
                             ))
    reg_opt = XGBRegressor(n_estimators=res_gp.x[0],
                            max_depth=res_gp.x[1],
                            num_parallel_tree=res_gp.x[2],
                            min_child_weight=res_gp.x[3],
                            learning_rate=res_gp.x[4],
                            subsample=res_gp.x[5],
                            gamma=res_gp.x[6],
                            alpha=res_gp.x[7],
                            reg_alpha=res_gp.x[8],
                            reg_lambda=res_gp.x[9]
                            )
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    if cif_stru == "origin":

        result = pd.ExcelWriter(cif_stru + "_" + "_gbr.xlsx")
    
        df_result_train = pd.DataFrame({"T": Xtrain["T"],
                                        "Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"T": Xtest["T"],
                                        "Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
    
        feature_important = reg_opt.get_booster().get_score(importance_type='weight')
        keys = ["PLD", "LCD", "VSA", "Density", "VF", "T"]
        values = list(feature_important.values())

        print("impact: ", keys, values)
    
        impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
        impact_fea.to_excel(result, index=False, sheet_name = "importance")

        result.close()
        save_model = joblib.dump(reg_opt,cif_stru + "_" + target +".pkl")

    else:

        result = pd.ExcelWriter(path + cif_stru + "_" + ratio + "_" + target +"_gbr.xlsx")
            
        df_result_train = pd.DataFrame({"T": Xtrain["T"],
                                        "Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"T": Xtest["T"],
                                        "Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
    
        values = []
        feature_importance = reg_opt.feature_importances_
        keys = ["PLD","LCD","AVSA","Density","VF","T","n-site","r-group"]
        feature_importance_dict = dict(zip(feature_names, feature_importance))
        for feature, importance in feature_importance_dict.items():
            print(f"{feature}: {importance}")
            values.append(importance)
    
        impact_fea.to_excel(result, index=False, sheet_name = "importance")

        result.close()
        save_model = joblib.dump(reg_opt,path + cif_stru + "_" + ratio + "_" + target + "_gbr.pkl")

def training_model_all(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = XGBRegressor()
    space  = [Integer(1,200, name='n_estimators'),
            Integer(1, 10, name='max_depth'),
            Integer(1, 10, name='num_parallel_tree'),
            Integer(1, 10, name='min_child_weight'),
            Real(0.001,1,"log-uniform",name='learning_rate'),
            Real(0.01,1,name='subsample'),
            Real(0.001,10,"log-uniform",name='gamma'),
            Real(0, 1, name='alpha'),
            Real(2, 10, name='reg_alpha'),
            Real(10, 50, name='reg_lambda')
         ]
    @use_named_args(space)

    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
        - n_estimator=%d
        - max_depth=%d
        - num_parallel_tree=%d
        - min_child_weight=%d
        - learning_rate=%f
        - subsample=%f
        - gamma=%f
        - alpha=%f
        - reg_alpha=%f
        - reg_lambda=%f""" % (res_gp.x[0],res_gp.x[1],
                            res_gp.x[2],res_gp.x[3],
                            res_gp.x[4],res_gp.x[5],
                            res_gp.x[6],res_gp.x[7],
                            res_gp.x[8],res_gp.x[9]
                             ))
    reg_opt = XGBRegressor(n_estimators=res_gp.x[0],
                            max_depth=res_gp.x[1],
                            num_parallel_tree=res_gp.x[2],
                            min_child_weight=res_gp.x[3],
                            learning_rate=res_gp.x[4],
                            subsample=res_gp.x[5],
                            gamma=res_gp.x[6],
                            alpha=res_gp.x[7],
                            reg_alpha=res_gp.x[8],
                            reg_lambda=res_gp.x[9]
                            )
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_gbr.xlsx")
    
    df_result_train = pd.DataFrame({"T": Xtrain["T"],
                                    "site":Xtrain["site"],
                                    "ratio":Xtrain["ratio"],
                                    "Ytrain": Ytrain.values.reshape(-1),
                                    'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
    df_result_train.to_excel(result, index=False, sheet_name = "data_train")
    df_result_test = pd.DataFrame({"T": Xtest["T"],
                                   "site":Xtest["site"],
                                   "ratio":Xtest["ratio"],
                                   "Ytest": Ytest.values.reshape(-1),
                                   'Ytest_pre': reg_opt.predict(Xtest).ravel()})
    df_result_test.to_excel(result, index=False, sheet_name = "data_test")
    
    values = []
    feature_importance = reg_opt.feature_importances_
    keys = ["PLD","LCD","AVSA","Density","VF","T","n-site","r-group"]
    feature_importance_dict = dict(zip(keys, feature_importance))
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")
        values.append(importance)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target +"_gbr.pkl")
