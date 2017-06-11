#-*- coding:utf8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from time import clock
import datetime



def read_data_sets(path, block_size = 200):
    X = []
    y = []
    _block_size = block_size
    count = 0
    round = 0
    with open(path) as f:
        start = clock()
        for sample in f:
            if sample:
                count += 1
                sample = sample.split(" ")
                target = sample[0].split(",")[1]  # 获取标签值
                y.append(float(target))
                sample[0] = sample[0].split(",")[2]
                sample = np.array(sample)
                sample = sample.astype("float32")
                # 处理
                X.append(sample)
                if count >= _block_size:
                    X, y = np.array(X).reshape(_block_size, -1), np.array(y).reshape(_block_size, -1)
                    round += 1
                    print("-----------get_sample {} time:{:.2f} s------{}------------\n".format(_block_size, clock()-start, datetime.datetime.now()))
                    yield X, y
                    start = clock()
                    X, y = [], []
                    count = 0



def extract_features(iter,savepath):

    file = open(savepath, 'ab')
    for i, (train, lables) in enumerate(iter):
        start = clock()
        print("------------iter{}---------------".format(i))
        batch_features = []
        train = train.reshape(batch_size,15,4,101,101)
        for s,sample in enumerate(train):
            print("------------sample{}---------------".format(s))
            last_time_map = sample[14,:,:,:]
            new_features = []
            # 1.最后一个时序的4个高度环范围反射率平均值特征,包括15个时序的变化统计特征：方差，极值，趋势，
            # 2.最后一个时序的4个高度局部范围反射率平均值特征，
            # 3.中心范围的特征20*20，,4*40,10*10,101*101
            # 4.统计各反射率频数
            # 5.选取中心小区域所有反射率作为特征
            # 6.局部的数据的方差
            # 7.高度之间的联系
            ring = []
            local = []
            centre = []
            frequency = []
            bit_map = []
            map_variance = []
            height_statistics = []
            height_set1 = []
            height_set2 = []
            height_set3 = []
            for h in range(0,4):
                # 15个时间点的一圈圈总量
                sum_set = []
                for t in range(15):
                    # 51圈中心总量
                    each_time_sum = []
                    for k in range(0, 51, centre_step):
                        each_time_sum.append(np.sum(sample[t, h, k:101 - k, k:101 - k]))
                    sum_set.append(each_time_sum)
                # 提取一环环的变化：
                ring_set = []
                for t in range(15):
                    each_time_ring = []
                    for a in range(0, 50, 2):
                        each_time_ring.append(
                            (sum_set[t][a] - sum_set[t][a + 2]) / ((101 - 2 * a) ** 2 - (101 - 2 * a - 4) ** 2))  # 25个环
                    ring_set.append(each_time_ring)
                # 加入最后一个时间点的环特征
                ring.extend(ring_set[14])
                # 提取15个时序环的统计特性，
                for a in range(25):
                    group = []
                    for t in range(15):
                        group.append(ring_set[t][a])
                    ring.append(np.var(group))
                    ring.append(np.mean(group))
                    ring.append(np.ptp(group))
                    ring.append(1 if group[14] >= np.max(group) else 0)
                    ring.append(np.max(group) - group[14])
                    ring.append(group[14] - np.min(group))
                    #斜率
                    A = np.vstack([np.arange(15), np.ones(15)]).T
                    m, c = np.linalg.lstsq(A, np.array(group))[0]
                    ring.append(m)
                    ring.append(c)

                centre_average_set = []
                for time in sample[:, h, 45:56, 45:56]:
                    centre_average_set.append(np.average(time))
                    map_variance.append(np.var(time))
                centre.append(centre_average_set[14])
                centre.append(np.var(centre_average_set))
                centre.append(np.mean(centre_average_set))
                centre.append(np.ptp(centre_average_set))
                centre.append(1 if centre_average_set[14] >= np.max(centre_average_set) else 0)
                centre.append(np.max(centre_average_set) - centre_average_set[14])
                centre.append(centre_average_set[14] - np.min(centre_average_set))

                #15个时序的斜率和截距
                A = np.vstack([np.arange(15), np.ones(15)]).T
                m, c = np.linalg.lstsq(A, np.array(centre_average_set))[0]
                centre.append(m)
                centre.append(c)

                centre_average_set = []
                for time in sample[:, h, 40:61, 40:61]:
                    centre_average_set.append(np.average(time))
                    map_variance.append(np.var(time))
                centre.append(centre_average_set[14])
                centre.append(np.var(centre_average_set))
                centre.append(np.mean(centre_average_set))
                centre.append(np.ptp(centre_average_set))
                centre.append(1 if centre_average_set[14] >= np.max(centre_average_set) else 0)
                centre.append(np.max(centre_average_set) - centre_average_set[14])
                centre.append(centre_average_set[14] - np.min(centre_average_set))

                # 15个时序的斜率和截距
                A = np.vstack([np.arange(15), np.ones(15)]).T
                m, c = np.linalg.lstsq(A, np.array(centre_average_set))[0]
                centre.append(m)
                centre.append(c)

                # 2.提取局部区域的平均反射率
                for step in [20,50,100]:
                    local_variance = []
                    for row in range(0, 101, step):
                        if row == 100:
                            break
                        for col in range(0, 101, step):
                            if col == 100:
                                break
                            local_area_reflection = np.average(
                                last_time_map[h, row:row + step, col:col + step])
                            local_variance.append(local_area_reflection)
                            local.append(local_area_reflection)
                            # 3.提取区域不同时序的方差（波动）10*10*4=400
                            local_sum_list = []
                            for time in sample[:, h, row:row + step, col:col + step]:
                                local_sum_list.append(np.average(time))
                            local.append(np.var(local_sum_list))
                            local.append(np.mean(local_sum_list))
                            local.append(np.ptp(local_sum_list))
                            centre.append(1 if local_sum_list[14] >= np.max(local_sum_list) else 0)
                            centre.append(np.max(local_sum_list) - local_sum_list[14])
                            centre.append(local_sum_list[14] - np.min(local_sum_list))
                            if step != 10:
                                A = np.vstack([np.arange(15), np.ones(15)]).T
                                m, c = np.linalg.lstsq(A, np.array(local_sum_list))[0]
                                local.append(m)
                                local.append(c)
                    map_variance.append(np.var(local_variance))

                # 增加20*20个点
                bit_map.extend(last_time_map[h, 40:61, 40:61].ravel())

                #增加五个特征4*5 = 20
                frequency.append(np.sum(last_time_map[h] <= 20))
                frequency.append(np.sum((last_time_map[h] > 20) & (last_time_map[h] <= 40)))
                frequency.append(np.sum((last_time_map[h] > 40) & (last_time_map[h] <= 60)))
                frequency.append(np.sum((last_time_map[h] > 60) & (last_time_map[h] <= 80)))
                frequency.append(np.sum((last_time_map[h] > 80) & (last_time_map[h] <= 100)))
                frequency.append(np.sum((last_time_map[h] > 100) & (last_time_map[h] <= 120)))
                frequency.append(np.sum((last_time_map[h] > 120) & (last_time_map[h] <= 140)))
                frequency.append(np.sum((last_time_map[h] > 140) & (last_time_map[h] <= 160)))
                frequency.append(np.sum((last_time_map[h] > 160)))

                # 最后一个时序高度的统计特性：
                height_set1.append(np.average(last_time_map[h, 45:56, 45:56]))
                height_set2.append(np.average(last_time_map[h, 40:61, 40:61]))
                height_set3.append(np.average(last_time_map[h]))
                # 最后一个时序高度的统计特性：
            for height_set in [height_set1,height_set2,height_set3]:
                height_statistics.append(np.mean(height_set))
                height_statistics.append(np.var(height_set))
                height_statistics.append(np.ptp(height_set))
                height_statistics.append(1 if height_set[3] >= np.max(height_set) else 0)
                height_statistics.append(np.max(height_set) - height_set[3])
                height_statistics.append(height_set[3] - np.min(height_set))

            # height_statistics.append(np.mean(height_set2))
            # height_statistics.append(np.var(height_set2))
            # height_statistics.append(np.ptp(height_set2))
            # height_statistics.append(1 if height_set2[3] >= np.max(height_set2) else 0)
            # height_statistics.append(np.max(height_set2) - height_set2[3])
            # height_statistics.append(height_set2[3] - np.min(height_set2))

            new_features.extend(lables[s])
            new_features.extend(ring)
            new_features.extend(centre)
            new_features.extend(bit_map)
            new_features.extend(map_variance)
            new_features.extend(local)
            new_features.extend(frequency)
            new_features.extend(height_statistics)
            batch_features.append(new_features)

        batch_features = np.array(batch_features)
        np.savetxt(file,batch_features,fmt='%.2f', delimiter=',')
        print("----------time={:.2f}-----------".format(clock()-start))
    file.close()

batch_size = 100
centre_step = 1
local_step1 = 10
local_step2 = 20
local_step3 = 50


train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"

svae_train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train10_3.csv"
svae_test_path =  "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test10_3.csv"

train_iter = read_data_sets(train_path,batch_size)
extract_features(train_iter,svae_train_path)
test_iter = read_data_sets(test_path,batch_size)
extract_features(test_iter,svae_test_path)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train10_3.csv"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test10_3.csv"
data = pd.read_csv(train_path,header=None)
# test_data = pd.read_csv(test_path,header=None)
# test = test_data.iloc[:,1:]

lable = data.iloc[:,0]
train = data.iloc[:,1:]

from sklearn.ensemble import GradientBoostingRegressor
n_estimators=[50,60,70,80,100]

for n in n_estimators:
    print("n_estimators={}".format(n))
    reg = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1,max_depth=2,random_state=0, loss='ls')
    reg.fit(train.iloc[2000:], lable.iloc[2000:])
    pred = reg.predict(train.iloc[:2000])
    rsme = np.sqrt(mean_squared_error(pred,lable.iloc[:2000]))
    print("test pred:{}".format(rsme))
    train_pred = reg.predict(train.iloc[2000:4000])
    train_rmse = np.sqrt(mean_squared_error(train_pred,lable.iloc[2000:4000]))
    print("train pred:{}".format(train_rmse))