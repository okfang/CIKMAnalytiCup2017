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
#
train_path = "../data/train_shuffle.txt"
test_path = "../data/test.txt"

# train_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\train_shuffle.txt"
# test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_testA\\data_new\\CIKM2017_testA\\testA.txt"

learning_rate = 0.001
training_iters = 10               #外循环次数
minibatch_size = 2               #批量梯度下降批大小
sample_size = 100                  #每次读取训练样本数
repeat_round = 10                #shuffle 次数

# RNN Network Parameters
rnn_n_hidden = 128 # hidden layer num of features


# <====================CNN=========================>

# tf Graph input
cnn_x1 = tf.placeholder(tf.float32, [None, 15,101, 101])
cnn_x2 = tf.placeholder(tf.float32, [None, 15,101, 101])
cnn_x3 = tf.placeholder(tf.float32, [None, 15,101, 101])
cnn_x4 = tf.placeholder(tf.float32, [None, 15,101, 101])


y = tf.placeholder(tf.float32, [None, 1])

# 卷积层运算
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
# 池化层运算
def maxpool2d(x, k = 2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

#自定义CNN模型, 需要同时生成15个结果
def CNN_15(x, weights, biases):
    #三层CNN,不使用全连接层，指示提取数据特征
    # Reshape input picture
    x = tf.unstack(x, 15, 1) # list 15 shape:minibatch_size*101*101
    result = []
    for t in x:
        # Convolution Layer7*7
        #101*101 —> 50*50
        t = tf.reshape(t, shape=[-1,101,101,1])
        conv1 = conv2d(t, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer5*5
        #51*51 —> 26*26
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Convolution Layer3*3
        # 25*25 —> 13*13
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=2)



        # # Fully connected layer
        # # Reshape conv2 output to fit fully connected layer input
        # fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        conv3 = tf.reshape(conv3,[-1, 13*13*64])

        result.append(conv3)

    # result sahpe: 15 * minibatch_size*64*12*12
    return result

#每一层卷积和池化层参数
cnn_weights1 = {
    'wc1': tf.Variable(tf.random_normal([7,7,1,16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 'wd1': tf.Variable(tf.random_normal([12*12*64, 1024])),
}
cnn_weights2 = {
    'wc1': tf.Variable(tf.random_normal([7,7,1,16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 'wd1': tf.Variable(tf.random_normal([12*12*64, 1024])),
}
cnn_weights3 = {
    'wc1': tf.Variable(tf.random_normal([7,7,1,16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 'wd1': tf.Variable(tf.random_normal([12*12*64, 1024])),
}
cnn_weights4 = {
    'wc1': tf.Variable(tf.random_normal([7,7,1,16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 'wd1': tf.Variable(tf.random_normal([12*12*64, 1024])),
}

cnn_biases1 = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
}
cnn_biases2 = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
}
cnn_biases3 = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
}
cnn_biases4 = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
}
# <=================RNN=======================>

def RNN1(x, weights, biases):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, rnn_n_steps, 1)

    #输入已经是rnn_n_steps * minibatch_size * feature的 形式了

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN"):
        lstm_cell = rnn.BasicLSTMCell(rnn_n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred  #shape:minibatch_size*1
def RNN2(x, weights, biases):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, rnn_n_steps, 1)

    #输入已经是rnn_n_steps * minibatch_size * feature的 形式了

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(rnn_n_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred  #shape:minibatch_size*1
def RNN3(x, weights, biases):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, rnn_n_steps, 1)

    #输入已经是rnn_n_steps * minibatch_size * feature的 形式了

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(rnn_n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred  #shape:minibatch_size*1
def RNN4(x, weights, biases):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, rnn_n_steps, 1)

    #输入已经是rnn_n_steps * minibatch_size * feature的 形式了

    # Define a lstm cell with tensorflow
    with tf.variable_scope("RNN",reuse=True):
        lstm_cell = rnn.BasicLSTMCell(rnn_n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return pred  #shape:minibatch_size*1

#define weight and biases
rnn_weights1 = {
    "out":tf.Variable(tf.random_normal([rnn_n_hidden,1]))
}
rnn_weights2 = {
    "out":tf.Variable(tf.random_normal([rnn_n_hidden,1]))
}
rnn_weights3 = {
    "out":tf.Variable(tf.random_normal([rnn_n_hidden,1]))
}
rnn_weights4 = {
    "out":tf.Variable(tf.random_normal([rnn_n_hidden,1]))
}

rnn_biases1 = {
    "out": tf.Variable(tf.random_normal([1]))
}
rnn_biases2 = {
    "out": tf.Variable(tf.random_normal([1]))
}
rnn_biases3 = {
    "out": tf.Variable(tf.random_normal([1]))
}
rnn_biases4 = {
    "out": tf.Variable(tf.random_normal([1]))
}

# 最後一層
weight_final_out = tf.Variable(tf.random_normal([4,1]))
biases_final_out = tf.Variable(tf.random_normal([1]))

#四个高度，四个cnn和rnn模型
cnn_out1 = CNN_15(cnn_x1, cnn_weights1, cnn_biases1)
cnn_out2 = CNN_15(cnn_x2, cnn_weights2, cnn_biases2)
cnn_out3 = CNN_15(cnn_x3, cnn_weights3, cnn_biases3)
cnn_out4 = CNN_15(cnn_x4, cnn_weights4, cnn_biases4)

rnn_temp1 = RNN1(cnn_out1, rnn_weights1, rnn_biases1)
rnn_temp2 = RNN2(cnn_out2, rnn_weights2, rnn_biases2)
rnn_temp3 = RNN3(cnn_out3, rnn_weights3, rnn_biases3)
rnn_temp4 = RNN4(cnn_out4, rnn_weights4, rnn_biases4)

#将4个高度的结果组合成minibatch_size*4,然后线性组合，结果的形式为：minibatch_size*1
rnn_integrate = tf.concat([rnn_temp1, rnn_temp2, rnn_temp3, rnn_temp4],1)
pred = tf.add(tf.matmul(rnn_integrate,weight_final_out ), biases_final_out)

# Define loss and optimizer
cost = tf.sqrt(tf.losses.mean_squared_error(y, pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #大循环，需要重新读取文件
    for i in range(training_iters):
        train_iterator = read_data_sets(train_path, sample_size)
        #因为内存不足，使用生成器逐块读取文件，每次读取sample_size个样本
        for k,(train, lables) in enumerate(train_iterator):
            # start = clock()
            #sample_size个样本进行shuffle,批量进行训练，每次minibatch_size个样本
            prem = np.arange(sample_size)
            for j in range(repeat_round):
                # start_round = clock()
                np.random.shuffle(prem)
                train, lables = train[prem], lables[prem]
                for z in range(0,sample_size, minibatch_size):
                    minibatch_train, minibatch_lables = train[z:z+minibatch_size], lables[z:z+minibatch_size]
                    # 将数据转化为标准形状minibatch_size*15*4*101*101
                    minibatch_train = minibatch_train.reshape([minibatch_size,15,4,101,101])
                    #将高度维抽取出来，分别进入4个cnn模型，每个高度的数据进入CNN_15网络后得到15*minibatch_size*features结果，进入rnn模型
                    minibatch_train = tf.unstack(minibatch_train, 4, 2) #shape minibatch_size*15*101*101ra
                    minibatch_train = sess.run(minibatch_train)
                    sess.run(optimizer, feed_dict={cnn_x1:minibatch_train[0], cnn_x2:minibatch_train[1], cnn_x3:minibatch_train[2], cnn_x4:minibatch_train[3], y:minibatch_lables})
                print("sample_size:{} minibatch_size:{} repeat_round:{}/{} {}\n".format(sample_size,minibatch_size,j, repeat_round, datetime.datetime.now()))
            #测试数据集的效果
            train = train.reshape([sample_size,15,4,101,101])
            train = tf.unstack(train, 4, 2)
            train = sess.run(train)
            score = sess.run(cost, feed_dict={cnn_x1:train[0], cnn_x2:train[1], cnn_x3:train[2], cnn_x4:train[3], y:lables})
            print("training_iters:{}/{}  round:{}/{} socre={} {}\n".format(i,training_iters, k, 10000/sample_size, score,datetime.datetime.now()))
        #每次外循环跑一次测试数据
        learning_rate *= 0.9
        result = []
        test_iterator = read_data_sets(test_path, sample_size)
        for m ,(minibatch_test, test_lables) in enumerate(test_iterator):
            minibatch_test = minibatch_test.reshape([sample_size, 15, 4, 101, 101])
            minibatch_test = tf.unstack(minibatch_test, 4, 2)
            minibatch_test = sess.run(minibatch_test)
            temp = sess.run(cost, feed_dict={cnn_x1:minibatch_test[0], cnn_x2:minibatch_test[1], cnn_x3:minibatch_test[3], cnn_x4:minibatch_test[4]})
            result.append(temp)
        result = np.array(result).reshape(2000,-1)
        # np.savetxt("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\result{}.csv".format(i), result,delimiter=',')
        np.savetxt("./results/result{}.csv".format(i),result, delimiter=',')

        #保存模型
        saver = tf.train.Saver()
        # saver.save(sess,"F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\models\\model{}.ckpt".format(i))
        saver.save(sess,"./models/model{}.ckpt".format(i))
    print("Optimization Finished!")
