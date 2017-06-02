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



def extract_features(iter):

    file = open("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_test.csv",'ab')
    for i, (train, lables) in enumerate(iter):

        print("------------iter{}---------------".format(i))
        batch_features = []
        train = train.reshape(batch_size,15,4,101,101)
        for s,sample in enumerate(train):
            print("------------sample{}---------------".format(s))
            last_time_map = sample[14,:,:,:]
            sample_new_features = []
            average_reflection_set = []
            sum_reflection_set  = []
            local_area_reflection_set = []
            #逐层提取特征
            for h in range(4):
                # 1.提取最后一个时序的中心范围总/平均反射率，有四层4*50*2==400
                for k in range(0,51,centre_step):
                    area_sum = np.sum(last_time_map[h,k:101-k,k:101-k])
                    area_average = np.average(last_time_map[h,k:101-k,k:101-k])
                    sum_reflection_set.append(area_sum)
                    average_reflection_set.append(area_average)
                #2.提取局部区域的平均反射率10*10*4
                for row in range(0,101,local_step):
                    if row == 100:
                        break
                    for col in range(0,101,local_step):
                        if col == 100:
                            break
                        local_area_reflection = np.average(last_time_map[h,row:row+local_step,col:col+local_step])
                        local_area_reflection_set.append(local_area_reflection)

            # print("sum:")
            # print(sum_reflection_set)
            # print("average:")
            # print(average_reflection_set)
            # print("local:")
            # print(local_area_reflection_set)

            sample_new_features.extend(lables[s])
            sample_new_features.extend(sum_reflection_set)
            sample_new_features.extend(average_reflection_set)
            sample_new_features.extend(local_area_reflection_set)
            batch_features.append(sample_new_features)
        batch_features = np.array(batch_features)
        np.savetxt(file,batch_features,fmt='%.2f', delimiter=',')
    file.close()

batch_size = 500
minibatch_size = 100
centre_step = 1
local_step = 10

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"

# train_iter = read_data_sets(train_path,batch_size)
# extract_features(train_iter)

test_iter = read_data_sets(test_path,batch_size)
extract_features(test_iter)