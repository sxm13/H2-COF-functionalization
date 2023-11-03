from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge,LinearRegression,Lasso,ElasticNet,HuberRegressor
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import numpy as np
import joblib
from ML.features5.data_prepare import data_prepare
import warnings
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import AdaBoostRegressor

import warnings

def training_model_ada(path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)

    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path=path, filename=filename, target=target)

    reg = AdaBoostRegressor()
    space = [
        Real(1e-6, 1, prior='log-uniform', name='learning_rate'),
        Integer(1, 200, name='n_estimators')
    ]

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
    print("- learning_rate=%.6f" % res_gp.x[0])
    print("- n_estimators=%.6f" % res_gp.x[1])

    reg_opt = AdaBoostRegressor(learning_rate=res_gp.x[0], n_estimators=int(res_gp.x[1]))

    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)),
                  mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                  np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_ada.xlsx")

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

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_ada.pkl")


def training_model_lr(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)

    reg_opt = LinearRegression()
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_lr.xlsx")
    
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
    
    feature_important = abs(reg_opt.coef_)
    keys = ["PLD", "LCD", "VSA", "Density", "VF", "T","site","ratio"]
    values = list(feature_important)

    print("impact: ", keys, values)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")
    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_lr.pkl")

def training_model_rr(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg_opt = Ridge()
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_rr.xlsx")
    
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
    
    feature_important = abs(reg_opt.coef_)
    keys = ["PLD", "LCD", "VSA", "Density", "VF", "T","site","ratio"]
    values = list(feature_important)

    print("impact: ", keys, values)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_rr.pkl")

def training_model_lasso(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = Lasso()
    space = [Real(1e-6, 1, prior='log-uniform', name='alpha')]
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
        - alpha=%.6f""" % (res_gp.x[0]))
    reg_opt = Lasso(alpha=res_gp.x[0])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_lasso.xlsx")
    
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
    
    feature_important = abs(reg_opt.coef_)
    keys = ["PLD", "LCD", "VSA", "Density", "VF", "T","site","ratio"]
    values = list(feature_important)

    print("impact: ", keys, values)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_lasso.pkl")

def training_model_en(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = ElasticNet()
    space = [Real(1e-6, 1, prior='log-uniform', name='alpha'),
            Real(0, 1, name='l1_ratio')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- alpha=%.6f" % res_gp.x[0])
    print("- l1_ratio=%.6f" % res_gp.x[1])
    
    reg_opt = ElasticNet(alpha=res_gp.x[0], l1_ratio=res_gp.x[1])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_en.xlsx")
    
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
    
    feature_important = abs(reg_opt.coef_)
    keys = ["PLD", "LCD", "VSA", "Density", "VF", "T","site","ratio"]
    values = list(feature_important)

    print("impact: ", keys, values)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_en.pkl")

def training_model_hbr(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = HuberRegressor()
    space = [Real(1, 100, prior='log-uniform', name='epsilon')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- epsilon=%.6f" % res_gp.x[0])
    
    reg_opt = HuberRegressor(epsilon=res_gp.x[0])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_hbr.xlsx")
    
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
    
    feature_important = abs(reg_opt.coef_)
    keys = ["PLD", "LCD", "VSA", "Density", "VF", "T","site","ratio"]
    values = list(feature_important)

    print("impact: ", keys, values)
    
    impact_fea = pd.DataFrame({"features": keys,
                                'score': values})
    
    impact_fea.to_excel(result, index=False, sheet_name = "importance")

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_hbr.pkl")

def training_model_kr(path = "./ML/features5/", filename = "ML_all", target = "wt", n_job = 4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path = path, filename = filename, target = target)
    
    reg = KernelRidge()
    space = [
        Real(1e-6, 1, prior='log-uniform', name='alpha'),
        Real(1e-6, 1, prior='log-uniform', name='gamma')
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
    print("Best parameters:")
    print("- alpha=%.6f" % res_gp.x[0])
    print("- gamma=%.6f" % res_gp.x[1])

    reg_opt = KernelRidge(alpha=res_gp.x[0], kernel='rbf', gamma=res_gp.x[1])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),
            mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),
            np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_kr.xlsx")
    
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
    
    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_kr.pkl")

def training_model_svr(path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path=path, filename=filename, target=target)
    
    reg = SVR()
    space = [Real(1e-6, 1, prior='log-uniform', name='C'),
             Real(1e-6, 1, prior='log-uniform', name='epsilon')]
    
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
    print("- C=%.6f" % res_gp.x[0])
    print("- epsilon=%.6f" % res_gp.x[1])
    
    reg_opt = SVR(C=res_gp.x[0], epsilon=res_gp.x[1])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)),
                  mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                  np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_svr.xlsx")
    
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

    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_svr.pkl")

def training_model_knn(path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    np.int = int
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path=path, filename=filename, target=target)
    
    reg = KNeighborsRegressor()
    space = [Integer(1, 20, name='n_neighbors'),
             Categorical(('uniform', 'distance'), name='weights')]
    
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
    print("- n_neighbors=%d" % res_gp.x[0])
    print("- weights=%s" % res_gp.x[1])
    
    reg_opt = KNeighborsRegressor(n_neighbors=res_gp.x[0], weights=res_gp.x[1])
    
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'
          .format(reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'
          .format(mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)),
                  mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'
          .format(np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                  np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

    result = pd.ExcelWriter(path + filename + "_" + target + "_knn.xlsx")
    
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
    
    result.close()
    save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_knn.pkl")

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
