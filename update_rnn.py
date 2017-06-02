#-*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from time import clock

def read_data_sets(path, batch_size = 200):
    X = []
    y = []
    _block_size = batch_size
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


def feature_change(train):
    # file = open("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\new_features\\rnn_feature.csv",
    #             'ab')
    batch_features = []
    train = train.reshape(batch_size, 15, 4, 101, 101)
    for s, sample in enumerate(train):

        # 逐层提取特征
        times = []
        for t in range(15):
            heights = []
            for h in range(4):
                # 1.提取最后一个时序的中心范围总/平均反射率，有四层4*50*2==400
                height = []
                for k in range(0, 51):
                    height.append(np.sum(sample[t,h, k:101 - k, k:101 - k]))
                    height.append(np.average(sample[t,h, k:101 - k, k:101 - k]))
                # 2.提取局部区域的平均反射率10*10*4
                for row in range(0, 101, local_step):
                    if row == 100:
                        break
                    for col in range(0, 101, local_step):
                        if col == 100:
                            break
                        height.append(np.average(sample[t,h, row:row + local_step, col:col + local_step]))
                #提取不同反射率范围的个数
                height.append(np.sum(sample[t,h,:,:] <=20))
                height.append(np.sum((sample[t, h, :, :] > 20) & (sample[t, h, :, :] <= 40)))
                height.append(np.sum((sample[t, h, :, :] > 40) & (sample[t, h, :, :] <= 60)))
                height.append(np.sum((sample[t, h, :, :] > 60) & (sample[t, h, :, :] <= 80)))
                height.append(np.sum(sample[t, h, :, :] > 80))

                heights.append(height)
            times.append(heights)
        batch_features.append(times)
    train = np.array(batch_features)
    return train



# Parameters
learning_rate = 0.001
training_iters = 10                #外循环次数
batch_size = 500
minibatch_size = 100                #批量梯度下降批大小
repeat_round = 200                  #shuffle 次数

# Network Parameters
n_input = 5+400+102# MNIST data input (img shape: 28*28)
n_steps = 15 # timesteps
n_hidden = 128 # hidden layer num of features

#feature change parameters
centre_step = 1
local_step = 5

# tf Graph input
x1 = tf.placeholder("float", [None, n_steps, n_input])
x2 = tf.placeholder("float", [None, n_steps, n_input])
x3 = tf.placeholder("float", [None, n_steps, n_input])
x4 = tf.placeholder("float", [None, n_steps, n_input])

y = tf.placeholder("float", [None,1])

#define weight
weights1 = {
    "out":tf.Variable(tf.random_normal([n_hidden,1]))
}
biases1 = {
    "out": tf.Variable(tf.random_normal([1]))
}
weights2 = {
    "out":tf.Variable(tf.random_normal([n_hidden,1]))
}
biases2 = {
    "out": tf.Variable(tf.random_normal([1]))
}
weights3 = {
    "out":tf.Variable(tf.random_normal([n_hidden,1]))
}
biases3 = {
    "out": tf.Variable(tf.random_normal([1]))
}
weights4 = {
    "out":tf.Variable(tf.random_normal([n_hidden,1]))
}
biases4 = {
    "out": tf.Variable(tf.random_normal([1]))
}


def RNN1(x, weights, biases):

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


def RNN2(x, weights, biases):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred


def RNN3(x, weights, biases):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred


def RNN4(x, weights, biases):
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred

# 最後一層
weight_final_out = tf.Variable(tf.random_normal([4,1]))
biases_final_out = tf.Variable(tf.random_normal([1]))

pred1 = RNN1(x1, weights1, biases1)
pred2 = RNN2(x2, weights2, biases2)
pred3 = RNN3(x3, weights3, biases3)
pred4 = RNN4(x4, weights4, biases4)

rnn_integrate = tf.concat([pred1, pred2, pred3, pred4],1)
pred = tf.add(tf.matmul(rnn_integrate,weight_final_out ), biases_final_out)


#define loss and optimizer
cost = tf.sqrt(tf.losses.mean_squared_error(y, pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

#window
train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"

result_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\rnn_result{}.csv"
model_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\models\\rnn_model{}.ckpt"

#linux
# train_path = "../data/train.txt"
# test_path = "../data/test.txt"
# result_path = "./results/rnn_result{}.csv"
# model_path= "./models/rnn_model{}.ckpt"


with tf.Session() as sess:
    sess.run(init)
    #大循环，需要重新读取文件
    for i in range(training_iters):
        train_iterator = read_data_sets(train_path, batch_size)
        #因为内存不足，使用生成器逐块读取文件，每次读取sample_size个样本
        for k,(train, lables) in enumerate(train_iterator):

            if k == 19:
                train = feature_change(train)
                train = train.reshape((batch_size, n_steps, 4,n_input))
                train = tf.unstack(train, 4, 2)
                train = sess.run(train)
                score = sess.run(cost, feed_dict={x1:train[0], x2:train[1],x3:train[2],x4:train[3],y:lables})
                print("final 500 sample score = {}".format(score))
                break

            start = clock()
            #sample_size个样本进行shuffle,批量进行训练，每次minibatch_size个样本

            #转换特征
            train = feature_change(train)

            prem = np.arange(batch_size)
            for j in range(repeat_round):
                start_round = clock()
                np.random.shuffle(prem)
                train, lables = train[prem], lables[prem]
                for z in range(0,batch_size, minibatch_size):
                    minibatch_train, minibatch_lables = train[z:z+minibatch_size], lables[z:z+minibatch_size]
                    minibatch_train = minibatch_train.reshape((minibatch_size, n_steps,4, n_input))
                    #分四层
                    minibatch_train = tf.unstack(minibatch_train,4,2)
                    minibatch_train = sess.run(minibatch_train)

                    sess.run(optimizer, feed_dict={x1:minibatch_train[0], x2:minibatch_train[1],x3:minibatch_train[2],x4:minibatch_train[3],y:minibatch_lables})
            train = train.reshape((batch_size, n_steps,4, n_input))
            train = tf.unstack(train, 4, 2)
            train = sess.run(train)
            score = sess.run(cost, feed_dict={x1:train[0], x2:train[1],x3:train[2],x4:train[3],y:lables})
            print("training_iters:{}/{}  round:{}/{} socre={} time={:.2f}s".format(i, training_iters,k,10000/batch_size, score, clock()-start))
        #每次外循环跑一次测试数据
        learning_rate *= 0.9
        result = []
        test_iterator = read_data_sets(test_path, batch_size)
        for m ,(minibatch_test, test_lables) in enumerate(test_iterator):
            minibatch_test = feature_change(minibatch_test)
            minibatch_test = minibatch_test.reshape((batch_size, n_steps,4, n_input))
            minibatch_test = tf.unstack(minibatch_test, 4, 2)
            minibatch_test = sess.run(minibatch_test)
            temp = sess.run(pred,feed_dict={x1: minibatch_test[0], x2: minibatch_test[1],x3: minibatch_test[2],x4: minibatch_test[3]})
            result.append(temp)
        result = np.array(result).reshape(2000,-1)
        np.savetxt(result_path.format(i), result,delimiter=',')

        #保存模型
        saver = tf.train.Saver()
        saver.save(sess,model_path.format(i))
    print("Optimization Finished!")

