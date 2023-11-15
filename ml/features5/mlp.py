from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np
import joblib
from ML.features5.data_prepare import data_prepare
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)
np.int = int
def data_pare(path="./ML/features5/", filename="ML_all", target="wt"):
    Xtrain, Xtest, Ytrain, Ytest = data_prepare(path=path, filename=filename, target=target)
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    joblib.dump(scaler, "scaler.gz")
    scaler = joblib.load("scaler.gz")
    Xtest = scaler.transform(Xtest)

    return Xtrain, Xtest, Ytrain, Ytest

def training_model_mlp_solver(Xtrain, Xtest, Ytrain, Ytest, path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    
    reg = MLPRegressor(random_state=44) 
    
    space = [Integer(10, 400, name='hidden_layer_sizes'),
             Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
             Categorical(['lbfgs','sgd', 'adam'], name='solver'),
             Real(1e-5, 1e-2, prior='log-uniform', name='alpha')]
    
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call,random_state=44)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- hidden_layer_sizes=%d" % res_gp.x[0])
    print("- solver=%s" % res_gp.x[2])

    reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = res_gp.x[2],alpha=res_gp.x[3],random_state=44)
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(
        reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(
        mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)), mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(
        np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
        np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

def training_model_mlp_sgd(Xtrain, Xtest, Ytrain, Ytest,path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    
    reg = MLPRegressor(random_state=66) 

    space = [Integer(10, 400, name='hidden_layer_sizes'),
             Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
             Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
             Categorical(['constant','invscaling',"adaptive"], name='learning_rate')]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
        print(result)
        return result
    
    res_gp = gp_minimize(objective, space, n_calls=call)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- learning_rate=%s" % res_gp.x[3])

    learning_rate_name = res_gp.x[3]
    
    if learning_rate_name == 'constant':
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                 Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                 Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
                 Real(1e-6, 1, prior='log-uniform', name='learning_rate_init'),
                 Integer(100, 500, name='max_iter'),
                 Categorical([True, False], name='warm_start'),
                 Real(0.1, 1, prior='uniform', name='momentum')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
            print(result)
            return result
    
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=66)

        print("Best score=%.4f" % res_gp.fun)
 
    
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "sgd", alpha=res_gp.x[2], learning_rate = "constant",
                           learning_rate_init = res_gp.x[3], max_iter = res_gp.x[4], warm_start=res_gp.x[5], momentum=res_gp.x[6],random_state=66)
        reg_opt.fit(Xtrain, Ytrain)
    
        print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(
                                            reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
        print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(
                                            mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)), mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
        print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(
                                            np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                                            np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))
    else:
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                 Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                 Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
                 Integer(100, 500, name='max_iter'),
                 Categorical([True, False], name='warm_start'),
                 Real(0.1, 1, prior='uniform', name='momentum')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
            print(result)
            return result
    
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=66)

        print("Best score=%.4f" % res_gp.fun)
    
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "sgd", alpha=res_gp.x[2], learning_rate = learning_rate_name,
                               max_iter = res_gp.x[3], warm_start=res_gp.x[4], momentum=res_gp.x[5],random_state=66)
        reg_opt.fit(Xtrain, Ytrain)

        print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(
                                            reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
        print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(
                                            mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)), mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
        print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(
                                            np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                                            np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))

    
    # result = pd.ExcelWriter(path + filename + "_" + target + "_mlp.xlsx")
    
    # df_result_train = pd.DataFrame({"T": Xtrain["T"],
    #                                 "site": Xtrain["site"],
    #                                 "ratio": Xtrain["ratio"],
    #                                 "Ytrain": Ytrain.values.reshape(-1),
    #                                 'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
    # df_result_train.to_excel(result, index=False, sheet_name="data_train")
    # df_result_test = pd.DataFrame({"T": Xtest["T"],
    #                                "site": Xtest["site"],
    #                                "ratio": Xtest["ratio"],
    #                                "Ytest": Ytest.values.reshape(-1),
    #                                'Ytest_pre': reg_opt.predict(Xtest).ravel()})
    # df_result_test.to_excel(result, index=False, sheet_name="data_test")
    
    # result.close()
    # save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_mlp.pkl")


def training_model_mlp_adam(Xtrain, Xtest, Ytrain, Ytest,path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    
    reg = MLPRegressor(random_state=88) 

    space = [Integer(10, 400, name='hidden_layer_sizes'),
             Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
             Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
             Real(1e-6, 1, prior='log-uniform', name='learning_rate_init'),
             Integer(100, 500, name='max_iter'),
             Categorical([True, False], name='warm_start'),
             Real(0.001, 0.999, prior='uniform', name='beta_1'),
             Real(0.001, 0.999, prior='uniform', name='beta_2'),
             Real(1e-12, 1e-1, prior='log-uniform', name='epsilon')]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
        print(result)
        return result
    
    res_gp = gp_minimize(objective, space, n_calls=call,random_state=88)

    print("Best score=%.4f" % res_gp.fun)

    reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "adam", alpha=res_gp.x[2],learning_rate_init=res_gp.x[3],
                               max_iter = res_gp.x[4], warm_start=res_gp.x[5],beta_1=res_gp.x[6],beta_2=res_gp.x[7], epsilon=res_gp.x[8],random_state=88)
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(
                                            reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(
                                            mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)), mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(
                                            np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
                                            np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))
       
    # result = pd.ExcelWriter(path + filename + "_" + target + "_adam_mlp.xlsx")
    
    # df_result_train = pd.DataFrame({"T": Xtrain["T"],
    #                                 "site": Xtrain["site"],
    #                                 "ratio": Xtrain["ratio"],
    #                                 "Ytrain": Ytrain.values.reshape(-1),
    #                                 'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
    # df_result_train.to_excel(result, index=False, sheet_name="data_train")
    # df_result_test = pd.DataFrame({"T": Xtest["T"],
    #                                "site": Xtest["site"],
    #                                "ratio": Xtest["ratio"],
    #                                "Ytest": Ytest.values.reshape(-1),
    #                                'Ytest_pre': reg_opt.predict(Xtest).ravel()})
    # df_result_test.to_excel(result, index=False, sheet_name="data_test")
    
    # result.close()
    # save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_adam_mlp.pkl")


def training_model_mlp_lbfgs(Xtrain, Xtest, Ytrain, Ytest,path="./ML/features5/", filename="ML_all", target="wt", n_job=4, call=100):
    
    reg = MLPRegressor(random_state=44) 
    
    space = [Integer(10, 400, name='hidden_layer_sizes'),
             Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
             Real(1e-5, 1e-2, prior='log-uniform', name='alpha')]
    
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
    
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call,random_state=44)

    print("Best score=%.4f" % res_gp.fun)
    print("Best parameters:")
    print("- hidden_layer_sizes=%d" % res_gp.x[0])

    reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "lbfgs",alpha=res_gp.x[2],random_state=44)
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(
        reg_opt.score(Xtrain, Ytrain), reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(
        mean_absolute_error(Ytrain, reg_opt.predict(Xtrain)), mean_absolute_error(Ytest, reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(
        np.sqrt(mean_squared_error(Ytrain, reg_opt.predict(Xtrain), squared=False)),
        np.sqrt(mean_squared_error(Ytest, reg_opt.predict(Xtest), squared=False))))
    
    # result = pd.ExcelWriter(path + filename + "_" + target + "_lbfgs_mlp.xlsx")
    
    # df_result_train = pd.DataFrame({"T": Xtrain["T"],
    #                                 "site": Xtrain["site"],
    #                                 "ratio": Xtrain["ratio"],
    #                                 "Ytrain": Ytrain.values.reshape(-1),
    #                                 'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
    # df_result_train.to_excel(result, index=False, sheet_name="data_train")
    # df_result_test = pd.DataFrame({"T": Xtest["T"],
    #                                "site": Xtest["site"],
    #                                "ratio": Xtest["ratio"],
    #                                "Ytest": Ytest.values.reshape(-1),
    #                                'Ytest_pre': reg_opt.predict(Xtest).ravel()})
    # df_result_test.to_excel(result, index=False, sheet_name="data_test")
    
    # result.close()
    # save_model = joblib.dump(reg_opt, path + filename + "_" + target + "_lbfgs_mlp.pkl")
