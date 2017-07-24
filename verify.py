import numpy as np
import sklearn as sk
import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
import xgboost as xgb

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train9.csv"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test9.csv"
data = pd.read_csv(train_path,header=None)
# test_data = pd.read_csv(test_path,header=None)
# test = test_data.iloc[:,1:]

lable = data.iloc[:,0]
train = data.iloc[:,1:]

# def rmse_cv(model,X,y):
#     rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)

# from sklearn.ensemble import GradientBoostingRegressor
# n_estimators=[100,150,200]
# max_depth = [2,3,4,5,6]
# for n in n_estimators:
#     print("n_estimators={}".format(n))
#     for d in max_depth:
#         print("max_depth={}".format(d))
#         reg = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1,max_depth=d,random_state=0, loss='ls')
#         reg.fit(train.iloc[2000:], lable.iloc[2000:])
#         pred = reg.predict(train.iloc[:2000])
#         rsme = np.sqrt(mean_squared_error(pred,lable.iloc[:2000]))
#         print("test pred:{}".format(rsme))
#         train_pred = reg.predict(train.iloc[2000:4000])
#         train_rmse = np.sqrt(mean_squared_error(train_pred,lable.iloc[2000:4000]))
#         print("train pred:{}".format(train_rmse))



# dtrain = xgb.DMatrix(train.iloc[2000:],label=lable.iloc[2000:])
reg = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1)
reg.fit(train.iloc[2000:], lable.iloc[2000:])
pred = reg.predict(train.iloc[:2000])
rsme = np.sqrt(mean_squared_error(pred, lable.iloc[:2000]))
print("test pred:{}".format(rsme))
train_pred = reg.predict(train.iloc[2000:4000])
train_rmse = np.sqrt(mean_squared_error(train_pred, lable.iloc[2000:4000]))
print("train pred:{}".format(train_rmse))