import numpy as np
import sklearn as sk
import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train2.csv"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test2.csv"
data = pd.read_csv(train_path,header=None)
test_data = pd.read_csv(test_path,header=None)
test = test_data.iloc[:,1:]

lable = data.iloc[:,0]
train = data.iloc[:,1:]

def rmse_cv(model,X,y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

from sklearn.ensemble import GradientBoostingRegressor
n_estimators=[100,130,150,170,180,200]

for n in n_estimators:
    print("n_estimators={}".format(n))
    reg = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1,random_state=0, loss='ls')
    reg.fit(train.iloc[2000:], lable.iloc[2000:])
    pred = reg.predict(train.iloc[:2000])
    rsme = np.sqrt(mean_squared_error(pred,lable.iloc[:2000]))
    print("test pred:{}".format(rsme))
    train_pred = reg.predict(train.iloc[2000:4000])
    train_rmse = np.sqrt(mean_squared_error(train_pred,lable.iloc[2000:4000]))
    print("train pred:{}".format(train_rmse))

    del(reg)
    reg_total = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1, random_state=0, loss='ls')
    reg_total.fit(train, lable)
    res = reg_total.predict(test)
    pd.DataFrame(res).to_csv(
        "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\gbdt_result_update2_{}.csv".format(n),
        header=False, index=False)

#
# reg = GradientBoostingRegressor(n_estimators=160, learning_rate=0.1, random_state=0, loss='ls')
#
# # rmse = rmse_cv(reg,train,lable)
# # print(rmse)
#
# reg.fit(train.iloc[2000:], lable.iloc[2000:])
# pred = reg.predict(train.iloc[:2000])
# rmse = np.sqrt(mean_squared_error(pred,lable.iloc[:2000]))
# print(rmse)
#
# res = reg.predict(test)
# pd.DataFrame(res).to_csv("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\gbdt_result_update2.csv",header=False,index=False)



