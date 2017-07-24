#-*- coding:utf8 -*-

import numpy as np
import pandas as pd
from time import clock
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import scipy.stats as st
import multiprocessing
import os

batch_size = 400
compare_size = 10
area_constrain = 10
area_size = 8
threshold = 0
topn = 5

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

def is_area_max(row, col, sample, size, threshold):
    rowStart = row - size if row > size else 0
    rowEnd = 100 if row + size > 100 else row + size
    colStart = col - size if col > size else 0
    colEnd = 100 if col + size > 100 else col + size
    if (sample[row, col] >= threshold) and (sample[row, col] == np.max(sample[rowStart:rowEnd, col])) and (
        sample[row, col] == np.max(sample[row, colStart:colEnd])):
        # print("val:{}".format(sample[row,col]))
        # print("row:{}".format(sample[rowStart:rowEnd,col]))
        # print("col:{}".format(sample[row,colStart:colEnd]))
        return True
    else:
        return False

def find_peeks(z):
    coordinates = []
    peeks = []
    for row in range(101):
        for col in range(101):
            if is_area_max(row, col, z, compare_size, threshold):
                if len(coordinates) > 0:
                    is_neighbour = False
                    skip = True
                    for c, coor in enumerate(coordinates):
                        if np.sqrt((row - coor[0]) ** 2 + (col - coor[1]) ** 2) < area_constrain:
                            is_neighbour = True
                            if z[row, col] >= peeks[c]:
                                # skip = False
                                # print("remove:{}  {}".format(coordinates[c], peeks[c]))
                                # print("change:{} {}".format((row, col), z[row, col]))
                                coordinates[c] = (row, col)
                                peeks[c] = z[row, col]
                                break
                    # if is_neighbour and skip:
                    #     print("skip {}  {}".format((row, col), z[row, col]))
                    if not is_neighbour:
                        # print("append:{} {}".format((row, col), z[row, col]))
                        coordinates.append((row, col))
                        peeks.append(z[row, col])
                else:
                    # print("append:{} {}".format((row, col), z[row, col]))
                    coordinates.append((row, col))
                    peeks.append(z[row, col])
    return np.array(coordinates), np.array(peeks),len(peeks)

def get_peek_area_average(coordinates,sample):
    """返回topn峰值区域的平均值"""
    area_sum = 0
    for coor in coordinates:
        row,col = coor[0],coor[1]
        rowStart = row - area_size if row > area_size else 0
        rowEnd = 100 if row + area_size > 100 else row + area_size
        colStart = col - area_size if col > area_size else 0
        colEnd = 100 if col + area_size > 100 else col + area_size
        area_sum += np.average(sample[rowStart:rowEnd,colStart:colEnd])
    return area_sum//len(coordinates )

def get_peek_area_sum(coordinates,sample):
    """返回所有峰值区域的总和"""
    area_sum = 0
    for coor in coordinates:
        row,col = coor[0],coor[1]
        rowStart = row - area_size if row > area_size else 0
        rowEnd = 100 if row + area_size > 100 else row + area_size
        colStart = col - area_size if col > area_size else 0
        colEnd = 100 if col + area_size > 100 else col + area_size
        area_sum += np.sum(sample[rowStart:rowEnd,colStart:colEnd])
    return area_sum

def get_distacne_info(coordinates,sample):
    """返回距离列表"""
    dist = []
    for coor in coordinates:
        row, col = coor[0], coor[1]
        dist.append(np.sqrt((row-50)**2+(col-50)**2))
    return np.array(dist)

def extract(lock,train,lables,save_path):
    batch_features = []
    print(train.shape[0])
    train = train.reshape(train.shape[0], 15, 4, 101, 101)
    for s, sample in enumerate(train):
        print("------process:{}------sample{}---------------".format(os.getpid(),s))
        new_features = []
        new_features.extend(lables[s])
        for h in range(0, 4):

            # 最后一个时序的特征
            peek_info = []
            area_info = []
            distance_info = []
            frequency = []
            centre_data = []
            time_sequence = []

            z = sample[14, h]
            coordinates, peeks, num = find_peeks(z)
            indexs = np.array(peeks).argsort()[-topn:]
            coordinates_topn = coordinates[indexs]
            peeks_topn = peeks[indexs]
            # 添加和峰值有关的特征
            peek_info.append(num)
            # peek_info.extend(peeks_topn)
            peek_info.append(np.max(peeks))
            peek_info.append(np.average(peeks))
            peek_info.append(np.std(peeks))
            # 添加峰值区域有关的特征
            area_info.append(get_peek_area_average(coordinates, z))
            # area_info.append(get_peek_area_sum(coordinates,z))
            # 添加距离特征
            distance_list = get_distacne_info(coordinates_topn, z)
            # distance_info.extend(distance_list)
            distance_info.append(np.std(distance_list))
            distance_info.append(np.average(distance_list))
            distance_info.append(
                np.sqrt((coordinates_topn[-1][0] - 50) ** 2 + (coordinates_topn[-1][1] - 50) ** 2))  # 最大点到中心的距离
            # 添加频率特征
            frequency.append(np.sum(z < 50))
            frequency.append(np.sum(z > 80))
            frequency.append(np.sum(z > 100))
            frequency.append(np.sum(z > 130))
            frequency.append(np.sum(z > 150))
            # 添加中心数据
            centre_data.append(np.average(z[40:61, 40:61]))
            centre_data.append(np.var(z[40:61, 40:61].ravel()))
            centre_data.append(np.average(z[45:56, 45:56]))
            centre_data.append(np.var(z[45:56, 45:56].ravel()))

            # 添加时序变化特征
            num_list = []  # 峰值点个数的变化
            maxpeek_dist_list = []  # 最大峰值点的距离变化
            average_dist_list = []  # 前5峰值点平均距离的变化
            variance_dist_list = []  # 前10峰值点的分布情况变化
            average_topn_peek_area_list = []  # 前5峰值区域的平均值变化
            total_average_list = []  # 整个区域的变化，判断是否涌入新的雨云
            for t in sample[:, h]:
                coordinates, peeks, num = find_peeks(t)
                indexs = np.array(peeks).argsort()[-topn:]
                coordinates_topn = coordinates[indexs]

                indexs2 = np.array(peeks).argsort()[-10:]
                coordinates_topn2 = coordinates[indexs2]

                total_average_list.append(np.average(t))
                num_list.append(num)
                average_topn_peek_area_list.append(get_peek_area_average(coordinates, t))
                maxpeek_dist_list.append(
                    np.sqrt((coordinates_topn[-1][0] - 50) ** 2 + (coordinates_topn[-1][1] - 50) ** 2))

                distance_list = get_distacne_info(coordinates_topn, t)
                average_dist_list.append(np.average(distance_list))

                distance_list = get_distacne_info(coordinates_topn2, t)
                variance_dist_list.append(np.std(distance_list))

            time_sequence.append(np.mean(num_list))
            time_sequence.append(1 if num_list[14] >= np.max(num_list) else 0)
            time_sequence.append(1 if num_list[14] >= num_list[0] else 0)

            time_sequence.append(np.std(maxpeek_dist_list))
            time_sequence.append(st.skew(maxpeek_dist_list))
            time_sequence.append(np.mean(maxpeek_dist_list))
            time_sequence.append(1 if maxpeek_dist_list[14] >= np.max(maxpeek_dist_list) else 0)
            time_sequence.append((maxpeek_dist_list[14] - maxpeek_dist_list[0]) / 15)
            time_sequence.append((maxpeek_dist_list[14] - maxpeek_dist_list[13]))

            time_sequence.append(np.std(average_dist_list))
            time_sequence.append(st.skew(average_dist_list))
            time_sequence.append(np.mean(average_dist_list))
            time_sequence.append(1 if average_dist_list[14] >= np.max(average_dist_list) else 0)
            time_sequence.append((average_dist_list[14] - average_dist_list[0]) / 15)
            time_sequence.append((average_dist_list[14] - average_dist_list[13]))

            time_sequence.append(np.mean(variance_dist_list))
            time_sequence.append(1 if variance_dist_list[14] >= np.max(variance_dist_list) else 0)
            time_sequence.append((variance_dist_list[14] - variance_dist_list[0]) / 15)
            time_sequence.append((variance_dist_list[14] - variance_dist_list[13]))

            time_sequence.append(np.std(average_topn_peek_area_list))
            time_sequence.append(st.skew(average_topn_peek_area_list))
            time_sequence.append(np.mean(average_topn_peek_area_list))
            time_sequence.append(1 if average_topn_peek_area_list[14] >= np.max(average_topn_peek_area_list) else 0)
            time_sequence.append((average_topn_peek_area_list[14] - average_topn_peek_area_list[0]) / 15)
            time_sequence.append((average_topn_peek_area_list[14] - average_topn_peek_area_list[13]))

            time_sequence.append(1 if total_average_list[14] >= np.max(total_average_list) else 0)
            time_sequence.append((total_average_list[14] - total_average_list[0]) / 15)
            time_sequence.append((total_average_list[14] - total_average_list[13]))

            new_features.extend(peek_info)
            new_features.extend(area_info)
            new_features.extend(distance_info)
            new_features.extend(frequency)
            new_features.extend(time_sequence)
            new_features.extend(centre_data)
        print("process:{}  shape:{}".format(os.getpid(),len(new_features)))
        batch_features.append(new_features)
    batch_features = np.array(batch_features)
    print("---process:{}------shape:{}-------------".format(os.getpid(),batch_features.shape))
    with lock:
        with open(save_path,"ab") as f:
            np.savetxt(f, batch_features, fmt='%.2f', delimiter=',')

# 提取2,3层的云图信息
# 提取每个时序的图的峰值点区域信息，
# 1.峰值点的个数
# 2.top10峰值点周围n*n区域的平均值和所有峰值区域的综合（表示总量）
# 3.峰值点的大小
# 4.最后一个时序峰值点到中心点的距离，所有峰值点到中心点的平均距离，15个时序平均距离的变化，最大峰值点的变化
# 5.top10峰值区域的平均值的变化
# 6.分段，100以上，130以上，150以上反射率的个数和变化
# 7.中心点区域数据
def extract_features(iter,savepath):
    for i, (train, lables) in enumerate(iter):
        start = clock()
        lock = multiprocessing.Lock()
        print("------------iter{}-------{}--------".format(i,datetime.datetime.now()))
        p1 = multiprocessing.Process(name="p1",target=extract, args=(lock,train[0:50],lables[0:50],savepath))
        p2 = multiprocessing.Process(name="p2",target=extract, args= (lock,train[50:100], lables[50:100], savepath))
        p3 = multiprocessing.Process(name="p3",target=extract, args = (lock,train[100:150], lables[100:150], savepath))
        p4 = multiprocessing.Process(name="p4",target=extract, args = (lock,train[150:200], lables[150:200], savepath))
        p5 = multiprocessing.Process(name="p5", target=extract, args=(lock, train[200:250], lables[200:250], savepath))
        p6 = multiprocessing.Process(name="p6", target=extract, args=(lock, train[250:300], lables[250:300], savepath))
        p7 = multiprocessing.Process(name="p7", target=extract, args=(lock, train[300:350], lables[300:350], savepath))
        p8 = multiprocessing.Process(name="p8", target=extract, args=(lock, train[350:400], lables[350:400], savepath))


        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()

        print("----------time={:.2f}----{}-------".format(clock() - start,datetime.datetime.now()))


# train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
# test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"
# save_train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train11.csv"
# save_test_path =  "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test11.csv"

train_path = "../data/train_shuffle.txt"
test_path = "../data/testB.txt"
save_train_path = "./new_feature/feature_update_trainB11.csv"
save_test_path = "./new_feature/feature_update_testB11.csv"


iterator = read_data_sets(train_path,batch_size)
extract_features(iterator,save_train_path)
test_iter = read_data_sets(test_path,batch_size)
extract_features(test_iter,save_test_path)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_train11.csv"
# test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_update_test11.csv"
data = pd.read_csv(save_train_path,header=None)
test_data = pd.read_csv(save_test_path,header=None)
test = test_data.iloc[:,1:]
print(data.shape)
lable = data.iloc[:,0]
train = data.iloc[:,1:]

from sklearn.ensemble import GradientBoostingRegressor
n_estimators=[50,100,150,180,200,300,]

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




# 提取对应的样本
# sample_list = []
# f = open(lable_path,"rb")
# lable = np.load(f)
# f.close()
# lable = pd.DataFrame(lable)
# sample_list.extend(lable[( lable.ix[:,0] == 0)].index.tolist()[:10])
# for i in range(0,100,10):
#     sample_list.extend(lable[( lable.ix[:,0] > i) & (lable.ix[:,0] <= i+10)].index.tolist()[:10])
# read_data_sets(train_path,sample_list)




# f = open(save_sample_path,"rb")
# sample = np.load(f)
# f.close()
# f = open(save_lable_path,"rb")
# lable = np.load(f)
# f.close()

# fig = plt.figure()
# ax = Axes3D(fig)

# z = sample[100,14,2]
# start = clock()
# compare_size = 10
# area_constrain = 10
# threshold = 0
# coordinates = []
# peeks = []
#
# for row in range(101):
#     for col in range(101):
#         if is_area_max(row,col,z,compare_size,threshold):
#             if len(coordinates) > 0:
#                 is_neighbour = False
#                 skip = True
#                 for c, coor in enumerate(coordinates):
#                     if np.sqrt((row - coor[0]) ** 2 + (col - coor[1])** 2) < area_constrain:
#                         is_neighbour = True
#                         if z[row, col] >= peeks[c]:
#                             skip = False
#                             print("remove:{}  {}".format(coordinates[c], peeks[c]))
#                             print("change:{} {}".format((row, col), z[row, col]))
#                             coordinates[c]=(row, col)
#                             peeks[c]= z[row, col]
#                             break
#                 if is_neighbour and skip:
#                     print("skip {}  {}".format((row, col), z[row, col]))
#                 if not is_neighbour:
#                     print("append:{} {}".format((row, col), z[row, col]))
#                     coordinates.append((row, col))
#                     peeks.append(z[row, col])
#
#             else:
#                 print("append:{} {}".format((row,col), z[row,col]))
#                 coordinates.append((row, col))
#                 peeks.append(z[row, col])
# print(coordinates)
# print(peeks)
# data = np.column_stack((np.array(coordinates),np.array(peeks)))
# indexs = np.array(peeks).argsort()[-10:]
# print(indexs)
# print(np.array(peeks)[indexs])
# print(np.array(coordinates)[indexs])
#
# for a in coordinates:
#     z[a[0],a[1]] = 255
# print("time:{}".format(clock()-start))


# x = np.arange(101)
# y = np.arange(101)
# X,Y= np.meshgrid(x,y)
# plt.imshow(z)
# plt.savefig("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\sample.png")
# ax.plot_surface(X,Y,z, rstride=1, cstride=1, cmap="rainbow")
# plt.show()








