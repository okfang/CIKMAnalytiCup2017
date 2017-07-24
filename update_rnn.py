#-*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from time import clock
import pandas as pd

# def read_data_sets(path, batch_size = 200):
#     X = []
#     y = []
#     _block_size = batch_size
#     count = 0
#     round = 0
#     with open(path) as f:
#         start = clock()
#         for sample in f:
#             if sample:
#                 count += 1
#                 sample = sample.split(" ")
#                 target = sample[0].split(",")[1]  # 获取标签值
#                 y.append(float(target))
#                 sample[0] = sample[0].split(",")[2]
#                 sample = np.array(sample)
#                 sample = sample.astype("float32")
#                 # 处理
#                 X.append(sample)
#                 if count >= _block_size:
#                     X, y = np.array(X).reshape(_block_size, -1), np.array(y).reshape(_block_size, -1)
#                     round += 1
#                     print("-----------get_sample {} time:{:.2f} s------------------".format(_block_size, clock()-start))
#                     yield X, y
#                     start = clock()
#                     X, y = [], []
#                     count = 0


# def feature_change(train):
#     # file = open("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\rnn_feature.csv",
#     #             'ab')
#     batch_features = []
#     train = train.reshape(batch_size, 15, 4, 101, 101)
#     for s, sample in enumerate(train):
#         # 逐层提取特征
#         times = []
#         for t in range(15):
#             time = []
#             ring = []
#             centre = []
#             local = []
#             map_variance = []
#             bit_map = []
#             frequency = []
#             for h in range(4):
#                 # 1.提取环平均反射率，有四层4*50*2==400
#                 sum_set = []
#                 for t in range(15):
#                     # 51圈中心总量
#                     each_time_sum = []
#                     for k in range(0, 51, centre_step):
#                         each_time_sum.append(np.sum(sample[t, h, k:101 - k, k:101 - k]))
#                     sum_set.append(each_time_sum)
#                 # 提取一环环的平均值：25环
#                 ring_set = []
#                 for t in range(15):
#                     each_time_ring = []
#                     for a in range(0, 50, 2):
#                         each_time_ring.append(
#                             (sum_set[t][a] - sum_set[t][a + 2]) / ((101 - 2 * a) ** 2 - (101 - 2 * a - 4) ** 2))  # 25个环
#                     ring_set.append(each_time_ring)
#                     # 15个时间点，4层高度，15*4*25=1500 特征
#                     ring.extend(each_time_ring)
#                 #2.提取中心范围
#                 centre.append(np.average(sample[t, h, 45:56, 45:56]))
#                 centre.append(np.average(sample[t, h, 40:61, 40:61]))
#                 centre.append(np.average(sample[t,h]))
#                 # 各点数据的方差
#                 map_variance.append(np.var(sample[t, h, 45:56, 45:56]))
#                 map_variance.append(np.var(sample[t, h, 40:61, 40:61]))
#
#                 #局部区域特征
#                 for step in [10,20, 50, 100]:
#                     local_average = []
#                     for row in range(0, 101, step):
#                         if row == 100:
#                             break
#                         for col in range(0, 101, step):
#                             if col == 100:
#                                 break
#                             local_average.append(np.average(sample[t,h, row:row + step, col:col + step]))
#                             local.append(np.average(sample[t,h, row:row + step, col:col + step]))
#                     local.append(np.var(local_average))
#
#                 #加入中间20*20个点
#                 bit_map.extend(sample[t,h, 40:61, 40:61].ravel())
#                 #计入频数
#                 frequency.append(np.sum(sample[t,h] <= 20))
#                 frequency.append(np.sum((sample[t,h] > 20) & (sample[t,h] <= 40)))
#                 frequency.append(np.sum((sample[t,h] > 40) & (sample[t,h] <= 60)))
#                 frequency.append(np.sum((sample[t,h] > 60) & (sample[t,h] <= 80)))
#                 frequency.append(np.sum((sample[t,h] > 80) & (sample[t,h] <= 100)))
#                 frequency.append(np.sum((sample[t,h] > 100) & (sample[t,h] <= 120)))
#                 frequency.append(np.sum((sample[t,h] > 120) & (sample[t,h] <= 140)))
#                 frequency.append(np.sum((sample[t,h] > 140) & (sample[t,h] <= 160)))
#                 frequency.append(np.sum((sample[t,h] > 160)))
#             time.extend(ring)
#             time.extend(centre)
#             time.extend(local)
#             time.extend(map_variance)
#             time.extend(bit_map)
#             time.extend(frequency)
#             times.append(time)
#         batch_features.append(times)
#     train = np.array(batch_features)
#     return train



# Parameters
learning_rate = 0.00005
training_iters = 100                #外循环次数
# batch_size = 20
minibatch_size = 100               #批量梯度下降批大小

# Network Parameters
n_input = 3856# MNIST data input (img shape: 28*28)
n_steps = 15 # timesteps
n_hidden = 256 # hidden layer num of features

#feature change parameters
centre_step = 1
local_step = 5

def RNN(x, weights, biases):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN"):
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred






#window
# train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
# test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"
#
# result_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\rnn_results\\rnn_result2{}.csv"
# model_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\rnn_models\\rnn_model2{}.ckpt"
#
# save_train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_rnn_train2.npy"
# save_test_path =  "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\feature_rnn_test2.npy"
#linux
train_path = "./feature_rnn_train2.npy"
test_path = "./feature_rnn_test2.npy"
lable_path = "./train_lable.npy"

result_path = "./results/rnn_result{}.csv"
model_path= "./models/rnn_model{}.ckpt"

train = []
with open(train_path,"rb") as f:
    train = np.load(f)
print(train.shape)
lable = []
with open(lable_path, "rb") as f:
    lable = np.load(f)
print(lable.shape)

test = []
with open(test_path,"rb") as f:
    test = np.load(f)






# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])

y = tf.placeholder("float", [None, 1])

# define weight
weights = {
    "out": tf.Variable(tf.random_normal([n_hidden, 1]))
}
biases = {
    "out": tf.Variable(tf.random_normal([1]))
}
pred = RNN(x, weights, biases)
#define loss and optimizer
cost = tf.sqrt(tf.losses.mean_squared_error(y, pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #大循环，需要重新读取文件
    train_train = train[:8000]
    train_lables = lable[:8000]
    for i in range(training_iters):
        start = clock()
        prem = np.arange(8000)
        np.random.shuffle(prem)
        train_train, train_lables = train_train[prem], train_lables[prem]
        for z in range(0,8000, minibatch_size):
            minibatch_train, minibatch_lables = train_train[z:z+minibatch_size], train_lables[z:z+minibatch_size]
            minibatch_train = minibatch_train.reshape((minibatch_size, n_steps, n_input))
            sess.run(optimizer, feed_dict={x:minibatch_train,y:minibatch_lables})
        train_rmse = sess.run(cost,feed_dict={x:train_train[:2000], y:train_lables[:2000]})
        verify_rsme = sess.run(cost, feed_dict={x: train[8000:], y: lable[8000:]})
        print("iter={},train pred:{}    verify pred:{}".format(i,train_rmse,verify_rsme))
        #每次外循环跑一次测试数据
        # learning_rate *= 0.9
    predicion = sess.run(pred,feed_dict={x: test})
    np.savetxt(result_path.format(i), predicion,delimiter=',')
    # #保存模型
    # saver = tf.train.Saver()
    # saver.save(sess,model_path.format(i))
    print("Optimization Finished!")

