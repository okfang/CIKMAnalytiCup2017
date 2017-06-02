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
            sample_new_features = []
            average_reflection_set = []
            local_area_reflection_set = []
            local_variance_set = []
            local_up_down_set = []
            reflection_count_set = []
            final_loop_set = []
            loop_up_down_set = []
            #逐层提取特征
            for h in range(4):
                # 1.提取最后一个时序的中心范围总/平均反射率，有四层4*50*2==400
                for k in range(0,51,centre_step):
                    area_average = np.average(last_time_map[h,k:101-k,k:101-k])
                    average_reflection_set.append(area_average)

                #15个时间点的一圈圈总量
                area_sum_15set = []
                for t in range(15):
                    #51圈中心总量
                    each_sum_set = []
                    for k in range(0, 51, centre_step):
                        each_sum_set.append(np.sum(sample[t,h,k:101-k,k:101-k]))
                    area_sum_15set.append(each_sum_set)

                #提取最后一个时间的一圈圈平均值特征：
                for a in range(0,50,2):
                    final_loop_set.append((area_sum_15set[14][a] - area_sum_15set[14][a+2])/(101-2*a)**2-(101-2*a-4)**2)

                #提取一圈圈的变化：
                up_down_15set = []
                for t in range(15):
                    each_time = []
                    for a in range(0, 50, 2):
                        each_time.append((area_sum_15set[t][a] - area_sum_15set[t][a+2]))
                    up_down_15set.append(each_time)
                for t in range(14):
                    for a in range(25):
                        loop_up_down_set.append(up_down_15set[t+1][a] - up_down_15set[t][a])

                #2.提取局部区域的平均反射率10*10*4=400
                for row in range(0,101,local_step1):
                    if row == 100:
                        break
                    for col in range(0,101,local_step1):
                        if col == 100:
                            break
                        local_area_reflection = np.average(last_time_map[h,row:row+local_step1,col:col+local_step1])
                        local_area_reflection_set.append(local_area_reflection)
                        # 3.提取区域不同时序的方差（波动）10*10*4=400
                        local_sum_list = []
                        for time in sample[:,h,row:row+local_step1,col:col+local_step1]:
                            local_sum_list.append(np.average(time))
                        local_variance_set.append(np.var(local_sum_list))
                # 2.提取局部区域的平均反射率10*10*4=400
                for row in range(0, 101, local_step2):
                    if row == 100:
                        break
                    for col in range(0, 101, local_step2):
                        if col == 100:
                            break
                        local_area_reflection = np.average(
                            last_time_map[h, row:row + local_step2, col:col + local_step2])
                        local_area_reflection_set.append(local_area_reflection)
                        # 3.提取区域不同时序的方差（波动）10*10*4=400
                        local_sum_list = []
                        for time in sample[:, h, row:row + local_step2, col:col + local_step2]:
                            local_sum_list.append(np.average(time))
                        local_variance_set.append(np.var(local_sum_list))
                # 2.提取局部区域的平均反射率10*10*4=400
                for row in range(0, 101, local_step2):
                    if row == 100:
                        break
                    for col in range(0, 101, local_step2):
                        if col == 100:
                            break
                        local_area_reflection = np.average(
                            last_time_map[h, row:row + local_step2, col:col + local_step2])
                        local_area_reflection_set.append(local_area_reflection)
                        # 3.提取区域不同时序的方差（波动）10*10*4=400
                        local_sum_list = []
                        for time in sample[:, h, row:row + local_step2, col:col + local_step2]:
                            local_sum_list.append(np.average(time))
                        local_variance_set.append(np.var(local_sum_list))
                # 2.提取局部区域的平均反射率10*10*4=400
                for row in range(0, 101, local_step4):
                    if row == 100:
                        break
                    for col in range(0, 101, local_step4):
                        if col == 100:
                            break
                        local_area_reflection = np.average(
                            last_time_map[h, row:row + local_step4, col:col + local_step4])
                        local_area_reflection_set.append(local_area_reflection)
                        # 3.提取区域不同时序的方差（波动）10*10*4=400
                        local_sum_list = []
                        for time in sample[:, h, row:row + local_step4, col:col + local_step4]:
                            local_sum_list.append(np.average(time))
                        local_variance_set.append(np.var(local_sum_list))

                #5.提取中心区域的反射量增减和方差,20*20 范围的中心区域 4*16=64
                centre_sum_list = []
                for time in sample[:,h,40:61,40:61]:
                    centre_sum_list.append(np.average(time))
                local_variance_set.append(np.var(centre_sum_list))

                for x in range(14):
                    local_up_down_set.append(np.average(sample[x+1, h, 40:61, 40:61]) - np.average(sample[x, h, 40:61, 40:61]))
                local_up_down_set.append(np.average(sample[14, h,40:61,40:61]) - np.average(sample[0, h,40:61,40:61]))
                #提取中心区域的反射量增减和方差,10*10 范围的中心区域 4*16=64
                centre_sum_list = []
                for time in sample[:, h,  45:56, 45:56]:
                    centre_sum_list.append(np.average(time))
                local_variance_set.append(np.var(centre_sum_list))

                for x in range(14):
                    local_up_down_set.append(
                        np.average(sample[x + 1, h,45:56, 45:56]) - np.average(sample[x, h, 45:56, 45:56]))
                local_up_down_set.append(
                    np.average(sample[14, h, 45:56, 45:56]) - np.average(sample[0, h, 45:56, 45:56]))

                #增加五个特征4*5 = 20
                reflection_count_set.append(np.sum(last_time_map[h] <= 20))
                reflection_count_set.append(np.sum((last_time_map[h] > 20) & (last_time_map[h] <= 40)))
                reflection_count_set.append(np.sum((last_time_map[h] > 40) & (last_time_map[h] <= 60)))
                reflection_count_set.append(np.sum((last_time_map[h] > 60) & (last_time_map[h] <= 80)))
                reflection_count_set.append(np.sum(last_time_map[h] > 80))

            # print("sum:")
            # print(sum_reflection_set)
            # print("average:")
            # print(average_reflection_set)
            # print("local:")
            # print(local_area_reflection_set)

            sample_new_features.extend(lables[s])
            # sample_new_features.extend(sum_reflection_set)
            sample_new_features.extend(average_reflection_set)
            sample_new_features.extend(local_area_reflection_set)
            sample_new_features.extend(local_variance_set)
            sample_new_features.extend(local_up_down_set)
            sample_new_features.extend(reflection_count_set)
            sample_new_features.extend(final_loop_set)
            sample_new_features.extend(loop_up_down_set)
            batch_features.append(sample_new_features)
        batch_features = np.array(batch_features)
        np.savetxt(file,batch_features,fmt='%.2f', delimiter=',')
        print("----------time={:.2f}-----------".format(clock()-start))
    file.close()

batch_size = 500
centre_step = 1
local_step1 = 10
local_step2 = 20
local_step3 = 50
local_step4 = 100

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"

svae_train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train3.csv"
svae_test_path =  "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test3.csv"

train_iter = read_data_sets(train_path,batch_size)
extract_features(train_iter,svae_train_path)

test_iter = read_data_sets(test_path,batch_size)
extract_features(test_iter,svae_test_path)