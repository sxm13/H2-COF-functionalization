from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np
import joblib
from ML.features5.data_prepare import data_prepare
import warnings
from sklearn.inspection import permutation_importance

def training_model_tree(path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path=path, filename=filename, target=target)
    
    reg = DecisionTreeRegressor()
    space = [Integer(1, 20, name='max_depth'),
             Integer(2, 30, name='min_samples_split'),
             Integer(1, 30, name='min_samples_leaf')]
    
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                           scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- max_depth=%d" % res_gp.x[0])
    print("- min_samples_split=%d" % res_gp.x[1])
    print("- min_samples_leaf=%d" % res_gp.x[2])
    
    reg_opt = DecisionTreeRegressor(max_depth=res_gp.x[0],
                                    min_samples_split=res_gp.x[1],
                                    min_samples_leaf=res_gp.x[2])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)),
                  mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                  np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_tree.xlsx")
    
    df_result_train = pd.DataFrame({"T": Xtrain["T"],
                                    "site": Xtrain["site"],
                                    "ratio": Xtrain["ratio"],
                                    "Ytrain": Ytrain.values.reshape(-1),
                                    'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
    df_result_train.to_excel(result, index=False, sheet_name="data_train")
    df_result_test = pd.DataFrame({"T": Xtest["T"],
                                   "site": Xtest["site"],
                                   "ratio": Xtest["ratio"],
                                   "Ytest": Ytest.values.reshape(-1),
                                   'Ytest_pre': reg_opt.predict(Xtest).ravel()})
    df_result_test.to_excel(result, index=False, sheet_name="data_test")

    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns

    print("impact: ", feature_names, feature_importance)
    
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "score": feature_importance})
    
    feature_importance_df.to_excel(result, index=False, sheet_name = "importance")
    
    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_tree.pkl")
