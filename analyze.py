#-*- coding:utf8 -*-

import numpy as np
import pandas as pd
from time import clock
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image
def read_data_sets(path, sample_list):
    X = []
    y = []
    print(sample_list)
    with open(path) as f:
        for i,sample in enumerate(f):
            if i in sample_list:
                print("index{}".format(i))
                sample = sample.split(" ")
                target = sample[0].split(",")[1]  # 获取标签值
                y.append(float(target))
                sample[0] = sample[0].split(",")[2]
                sample = np.array(sample)
                sample = sample.astype("float32")
                # 处理
                X.append(sample)
    X = np.array(X).reshape((len(sample_list),15,4,101,101))
    y = np.array(y).reshape((len(sample_list),1))
    print(X.shape)
    print(y.shape)
    with open("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\sample_list.npy","wb") as f:
        np.save(f, X)
    with open("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\sample_lable.npy","wb") as f:
        np.save(f,y)





def extract_features(iter,savepath):
    pass

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"
save_sample_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\sample_list.npy"
save_lable_path =  "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\sample_lable.npy"

lable_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\train_lable.npy"


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

f = open(save_sample_path,"rb")
sample = np.load(f)
print(sample.shape)
f.close()
f = open(save_lable_path,"rb")
lable = np.load(f)
f.close()
print(lable.shape)
print(len(lable))

for i in [0,10,20,30,40,50,60,70]:
    print(i)
    rpic = []
    for t in range(15):
        hpic = []
        for h in range(4):
            hpic.append(sample[i,t,3-h])
        rpic.append(np.row_stack(hpic))
    pic = np.column_stack(rpic)
    plt.title(lable[i])
    plt.imshow(pic)
    plt.show()
    # plt.savefig('F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\sample_picture\\sample_{}_{}.png'.format(lable[i],i))
    # im = Image.fromarray(np.uint8(pic))
    # im = im.convert('P')
    # im.show()
    # im.save('F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\sample_picture\\sample_{}_{}.png'.format(lable[i],i))







