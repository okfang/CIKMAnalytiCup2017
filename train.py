import numpy as np
import sklearn as sk
import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
# import xgboost as xgb

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train10_5.csv"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_testB10_5.csv"


data = pd.read_csv(train_path,header=None)
data = data.as_matrix()
lable = data[:,0]
train = data[:,1:]
# train = preprocessing.scale(train)

test = pd.read_csv(test_path,header=None)
test = test.as_matrix()
test = test[:,1:]
# test = preprocessing.scale(test)


# reg = MLPRegressor(hidden_layer_sizes=(100,),max_iter=10000,activation="relu")
# reg = svm.SVR()
# reg = linear_model.LinearRegression()
reg = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.1)
reg.fit(train[2000:], lable[2000:])
print("velify:")
pred = reg.predict(train[:2000])
rmse = np.sqrt(mean_squared_error(pred,lable[:2000]))
result = np.column_stack((pred,lable[:2000]))
# pd.DataFrame(result).to_csv("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\check_{}_{}.csv".format(d,n),header=False,index=False)
print(rmse)
print("train:")
pred = reg.predict(train[2000:4000])
rmse = np.sqrt(mean_squared_error(pred,lable[2000:4000]))
print(rmse)
# for d in [5,]:
#     for n in [5000]:
#         # reg = linear_model.ElasticNet(alpha=0.001,max_iter=100000)
#         # reg = svm.SVR()·
#         for r in [0.01]:
#             print("max_depth={} estimators={} learning_rate={}".format(d,n,r))
#             reg = GradientBoostingRegressor(n_estimators=n, max_depth=d, learning_rate=r)
#             reg.fit(train[2000:], lable[2000:])
#             print("velify:")
#             pred = reg.predict(train[:2000])
#             rmse = np.sqrt(mean_squared_error(pred,lable[:2000]))
#             result = np.column_stack((pred,lable[:2000]))
#             # pd.DataFrame(result).to_csv("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\check_{}_{}.csv".format(d,n),header=False,index=False)
#             print(rmse)
#             print("train:")
#             pred = reg.predict(train[2000:4000])
#             rmse = np.sqrt(mean_squared_error(pred,lable[2000:4000]))
#             print(rmse)

reg = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.1)
# reg = svm.SVR()
reg.fit(train, lable)
res = reg.predict(test)
pd.DataFrame(res).to_csv("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\gbdt_result_updateB10_5.csv",header=False,index=False)



