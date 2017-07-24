import numpy as np
import pandas as pd
from time import clock
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

def preprocess_data(path, minibatch_size=50):
    X = []
    y = []
    countNum = 0
    with open(path) as f:
        for sample in f:
            if sample:

                start = clock()

                countNum += 1
                numMap = sample.split(" ")
                target = numMap[0].split(",")[1]  # 获取标签值
                y.append(float(target))
                numMap[0] = numMap[0].split(",")[2]
                numMap = np.array(numMap)
                numMap = numMap.astype("float32")
                numMap = np.reshape(numMap, [15, 4, 101, 101])
                # 处理
                time = 0
                newMap = np.zeros([15, 4, 51])
                for a, t in enumerate(numMap):
                    time += 1
                    for b, h in enumerate(t):
                        i = 0
                        while i <= 50:
                            count = 0
                            total = 0
                            count += sum(h[i, i:101 - i] > 0)
                            top = sum(h[i, i:101 - i][h[i, i:101 - i] > 0])
                            count += sum(h[100 - i, i:101 - i] > 0)
                            bottom = sum(h[100 - i, i:101 - i][h[100 - i, i:101 - i] > 0])
                            count += sum(h[i + 1:101 - i - 1, i] > 0)
                            left = sum(h[i + 1:101 - i - 1, i][h[i + 1:101 - i - 1, i] > 0])
                            count += sum(h[i + 1:101 - i - 1, 100 - i] > 0)
                            right = sum(h[i + 1:101 - i - 1, 100 - i][h[i + 1:101 - i - 1, 100 - i] > 0])
                            total += top + bottom + left + right
                            newMap[a, b, i] = (total // count)
                            i += 1
                            #                     print("************************[time:{}]**************************".format(time))

                X.append(newMap.ravel())
                if countNum >= minibatch_size:
                    X, y = np.array(X), np.array(y)
                    print("---------------------pre used time:{}--------------------".format(clock() - start))
                    yield X, y
                    X, y = [], []
                    countNum = 0

def rmse(pred, y):
    return np.sqrt(mean_squared_error(pred, y))

if __name__ == "__main__":

    train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train.txt"
    minibatchs_iterator = preprocess_data(train_path, 200)

    sgd_clf = SGDRegressor()
    X_test = []
    y_test = []

    for i, (X_train, y_train) in enumerate(minibatchs_iterator):
        if i < 40:
            start = clock()
            sgd_clf.partial_fit(X_train, y_train)
            print("--------train{}-----------".format(i))
            print("------------train time{}---------".format(clock() - start))
        elif i < 50:  # 测试数据集
            X_test.append(X_train)
            y_test.append(y_train)
            print("--------test{}-----------".format(i))
        else:
            break
    X_test = np.array(X_test).reshape(2000, -1)
    y_test = np.array(y_test).reshape(2000, -1)
    print(X_test.shape)
    print(y_test.shape)
    pred = sgd_clf.predict(X_test)
    score = rmse(pred, y_test)
    print(score)