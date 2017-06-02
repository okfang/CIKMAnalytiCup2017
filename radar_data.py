#-*- coding: utf8 -*-

import numpy as np
from time import clock

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
                    print("-----------get_sample {} time:{:.2f} s------------------".format(_block_size, clock()-start))
                    yield X, y
                    start = clock()
                    X, y = [], []
                    count = 0


# if __name__ == "__main__":
#     path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train.txt"
#     data_iter = read_data_sets(path,10)
#     train_batch = []
#     lable_batch = []
#     for i in range(2):
#         X, y = next(data_iter)
#         print(X)
#         print(y)
#         train_batch.append(X)
#         lable_batch.append(y)
#     print(len(train_batch))