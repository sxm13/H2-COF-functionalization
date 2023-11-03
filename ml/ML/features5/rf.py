from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import joblib
from ML.features5.data_prepare import data_prepare
import warnings


def training_model_all(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = RandomForestRegressor()
    space  = [Integer(1, 200, name='n_estimators'),
              Integer(1, 30, name='max_depth'),
              Integer(2, 30, name='min_samples_split'),
              Integer(1, 30, name='min_samples_leaf'),
              Integer(1, 300, name='random_state')
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
    - min_samples_split=%d
    - min_samples_leaf=%d
    - random_state=%d""" % (res_gp.x[0], res_gp.x[1],
                         res_gp.x[2], res_gp.x[3],
                         res_gp.x[4]
                         ))
    reg_opt = RandomForestRegressor(n_estimators=res_gp.x[0],
                                max_depth=res_gp.x[1],
                                min_samples_split=res_gp.x[2],
                                min_samples_leaf=res_gp.x[3],                            
                                random_state=res_gp.x[4])
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_rf.xlsx")
    
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
    
    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns

    print("impact: ", feature_names, feature_importance)
    
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    
    feature_importance_df.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_rf.pkl")
