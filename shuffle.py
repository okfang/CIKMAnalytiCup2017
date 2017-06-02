import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from time import clock
import linecache

train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"


newpath = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"

def shuffle():
    print(linecache.getline(train_path, 1))


newfile = open(newpath, 'a')
for i in range(100):
    count = 0
    start = clock()
    print("======{}======".format(i))
    with open(train_path) as f:
        for x,line in enumerate(f):
            if line:
                if x < i:
                    continue
                elif x == i:
                    print(x)
                    newfile.write(line)
                count += 1
                if count > 100:
                    count = 1
                    print(x)
                    newfile.write(line)
        print("======time{:.2f}======".format(clock()-start))
newfile.close()

