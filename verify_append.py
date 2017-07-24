import numpy as np
import sklearn as sk
import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
import pickle
from sklearn.ensemble import RandomForestRegressor
train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train7.csv"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test7.csv"
append_test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\texture_test_dict.txt"
append_train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\texture_dict.txt"

data = pd.read_csv(train_path,header=None)
# test_data = pd.read_csv(test_path,header=None)
# test = test_data.iloc[:,1:]
file = open(append_train_path,"rb")
append_train_feature = pickle.load(file)
append_train_feature = pd.DataFrame.from_dict(append_train_feature,orient="index",dtype="float32")

print(append_train_feature.shape)
file.close()

file = open(append_test_path,"rb")
append_test_feature = pickle.load(file)
append_test_feature = pd.DataFrame.from_dict(append_test_feature,orient="index",dtype="float32")
file.close()

append_train_feature.fillna(0,inplace=True)
print(append_train_feature.isnull().any())

print(data.shape)
data = np.concatenate((data, append_train_feature),axis=1)

print(data.shape)

lable = data[:,0]
train = data[:,1:]

# def rmse_cv(model,X,y):
#     rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)

from sklearn.ensemble import GradientBoostingRegressor
n_estimators=[100,150,200]
rate = [0.1]
for n in n_estimators:
    print("n_estimators={}".format(n))
    for r in rate:
        print("learning_rate={}".format(r))
        reg = GradientBoostingRegressor(n_estimators=n, learning_rate=r,random_state=0, loss='ls')
        reg.fit(train[2000:], lable[2000:])
        pred = reg.predict(train[:2000])
        rsme = np.sqrt(mean_squared_error(pred,lable[:2000]))
        print("test pred:{}".format(rsme))
        train_pred = reg.predict(train[2000:4000])
        train_rmse = np.sqrt(mean_squared_error(train_pred,lable[2000:4000]))
        print("train pred:{}".format(train_rmse))

# reg = GradientBoostingRegressor(n_estimators=98, learning_rate=0.1,random_state=0, loss='ls')
# reg.fit(train.iloc[2000:], lable.iloc[2000:])
# pred = reg.predict(train.iloc[:2000])
# rsme = np.sqrt(mean_squared_error(pred, lable.iloc[:2000]))
# print("test pred:{}".format(rsme))
# train_pred = reg.predict(train.iloc[2000:4000])
# train_rmse = np.sqrt(mean_squared_error(train_pred, lable.iloc[2000:4000]))
# print("train pred:{}".format(train_rmse))