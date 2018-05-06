# -*- coding:utf-8 -*-
from __future__ import division
import time, datetime
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
INPUT_NODE = 7
OUTPUT_NODE = 1
LAYER1_NODE = 100
LAYER2_NODE = 10
BATCH_SIZE = 1
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 15
MOVING_AVERAGE_DECAY = 0.99
# mnist = input_data.read_data_sets("/home/zhp/PycharmProjects/tensorflow/mnist", one_hot=True)
N_BATCH = 10
def dataloadinput(address1):
    array3 = []
    with open(address1,'r') as lines:
        for line in lines:
            array2 = line.strip().split()
            array3.append(array2)
    return array3
def dataload(address):
    array = []
    array1 = []
    with open(address, 'r') as lines:
        for line in lines:
            array.append(line)
            lable = line.strip().split()
            array1.append(lable)
    dict1 = {}
    flavor = []
    for i in range(len(array)):
        flavora = array[i].strip().split()[1]
        if flavora not in flavor:
            flavor.append(flavora)
    for j in range(len(flavor)):
        lista = []
        lista1 = []
        for i in range(len(array)):
            flavorb = array[i].strip().split()[1]
            date = array[i].strip().split()[2]
            time = array[i].strip().split()[3]
            xingqi = datetime.datetime(int(array1[i][2].strip().split('-')[0]), int(array1[i][2].strip().split('-')[1]),
                                       int(array1[i][2].strip().split('-')[2]))
            xingqi1 = xingqi.weekday()
            # print flavorb
            # print flavor[j]
            if flavorb in flavor[j]:
                lista.append([date, xingqi1 + 1])
                lista1 = lista[:]
        dict1[flavor[j]] = lista
    dict2 = {}
    for flavor_name in dict1.keys():
        count = 0
        x = []
        y = []
        flag = dict1[flavor_name][0][0]
        for a in dict1[flavor_name]:
            if a[0] == flag:
                count +=1
                x = [a[0], a[1], count]
            else:
                flag = a[0]
                y.append(x)
                count = 1
                x = [a[0], a[1], count]
        #if dict1[flavor_name][len(dict1[flavor_name])-1] == dict1[flavor_name][len(dict1[flavor_name])-2]:
        y.append(x)
        dict2[flavor_name] = y
    dict3 = {}
    line1 = []
    ii = []
    for frame_name1 in dict2.keys():
        line1 = dict2[frame_name1][:]
        qq = len(line1)
        j1 = line1[0][0].strip().split('-')
        if j1[1] in ['01', '03', '05', '07', '08', '10', '12']:
            DATE_NUM = 32
        elif j1[1] in ['02']:
            DATE_NUM = 29
        else:
            DATE_NUM = 31
        jj = int(j1[2])
        for i1 in range(qq):
            ii.append(int(line1[i1][0].strip().split('-')[2]))
        for jj1 in range(1, DATE_NUM):     # range(jj, DATE_NUM)
                    if jj1 not in ii:
                        if jj1 <= 9:
                            line1.append([j1[0] + '-' + j1[1] + '-' + '0' + str(jj1), 0, 0])
                        else:
                            line1.append([j1[0]+'-'+j1[1]+'-'+str(jj1), 0, 0])
        dict3[frame_name1] = line1
        line1 = []
        ii = []
    dict4 = {}
    linee = []
    for frame_name1 in dict3.keys():
        linee = dict3[frame_name1][:]
        linee.sort(key=lambda x: x[0])
        dict4[frame_name1] = linee
    oo = ['flavor1', 'flavor2', 'flavor3', 'flavor4', 'flavor5', 'flavor6', 'flavor7', 'flavor8', 'flavor9', 'flavor10','flavor11', 'flavor12', 'flavor13', 'flavor14', 'flavor15']
    jjj = []
    for i in range(1, 34):
        jjj.append(['2015-00-00', 0, 0])
    for frame_name2 in oo:
        if frame_name2 not in dict4.keys():
            dict4[frame_name2] = jjj
    dict5 = {}
    dict6 = {}
    lastline = []
    lastline1 = []
    label = []
    qqq = 0
    jkj = 0
    for frame_name3 in dict4.keys():
        kkk = (len(dict4[frame_name3])) // 7 * 7
        for ill in range(1, kkk-7):
            for jjj in range(7):
                lastline.append(dict4[frame_name3][ill - 1+jjj][2])
            lastline1.append([lastline])
            lastline = []
            if ill + 14 <= kkk:
                jkj = ill
                for ooo in range(jkj+7, jkj + 14):
                    aaaa = dict4[frame_name3][ooo][2]
                    qqq = qqq + aaaa
            label.append([[qqq]])
            qqq = 0
            # if ill % 7 == 0:
            #    lastline1.append([lastline])
            #    lastline = []
            #    if ill+7 <= kkk:
            #        jkj = ill
            #        for ooo in range(jkj, jkj+7):
            #            aaaa = dict4[frame_name3][ooo][2]
            #            qqq = qqq + aaaa
            #    label.append([[qqq]])
            #    qqq = 0
        dict5[frame_name3] = lastline1
        dict6[frame_name3] = label
        label = []
        lastline = []
        lastline1 = []
    return dict4, dict5, dict6
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2, weights3, biases3 ):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
        layer3 = tf.nn.relu(tf.matmul(layer2, weights3) + biases3)
        return layer3
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(biases3))
        return layer3
def main():
    a, a1, a2 = dataload("/home/zhp/PycharmProjects/huawei/1月份数据.txt")
    b, b1, b2 = dataload("/home/zhp/PycharmProjects/huawei/2月数据.txt")
    # c, c1, c2 = dataload("/home/zhp/PycharmProjects/huawei/")
    flage = True
    while (flage):
        print("please input a predicated day or shutdown for entering d:")
        predicated_day = raw_input()
        if predicated_day == 'd':
            flage = False
        else:
            print a
            print a1
            print a2
            print b
            print b1
            print b2
            x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
            weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
            biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
            weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
            biases2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
            weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
            biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
            y = inference(x, None, weights1, biases1, weights2, biases2, weights3, biases3)
            global_step = tf.Variable(0, trainable=False)
            variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_average.apply(tf.trainable_variables())
            average_y = inference(x, variable_average, weights1, biases1, weights2, biases2, weights3, biases3)
            cross_entropy = tf.nn.l2_loss(y-y_)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
            regularization = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
            loss = cross_entropy_mean + regularization
            learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                       10, LEARNING_RATE_DECAY)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            train_op = tf.group(train_step, variables_averages_op)
            correct_prediction = tf.sqrt(tf.squared_difference(y_, y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            saver = tf.train.Saver()
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                test_feed = {x: np.array(b1['flavor11']), y_: np.transpose(np.array(b2['flavor11']))}
                for epoch in range(TRAINING_STEPS):
                        batch_xs = np.array(a1['flavor11'][epoch])
                        batch_ys = np.transpose(np.array(a2['flavor11'][epoch]))
                        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
                A = np.mat(np.array(a1['flavor11']))
                train_acc = sess.run(accuracy, feed_dict={x: A, y_: np.transpose(np.mat(np.array(a2['flavor11'])))})
                print("After %d training steps,train accuracy using average model is %g" % (epoch, train_acc))
                print np.transpose(np.mat(np.array(a2['flavor11'])))
                for epoch1 in range(17):
                    test_acc = sess.run(accuracy, feed_dict={x: np.array(b1['flavor11'][epoch1]), y_: np.transpose(np.array(b2['flavor11'][epoch1]))})
                    # print sess.run(average_y)
                    print("After %d test steps,test accuracy using average model is %g" % (epoch1, test_acc))
                    print sess.run(y, feed_dict={x: np.array(b1['flavor11'][epoch1]), y_: np.transpose(np.array(b2['flavor11'][epoch1]))})
                    print sess.run(y_, feed_dict={x: np.array(b1['flavor11'][epoch1]), y_: np.transpose(np.array(b2['flavor11'][epoch1]))})
if __name__ == '__main__':
    main()
'''import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
INPUT_NODE = 7
OUTPUT_NODE = 1
LAYER1_NODE = 500
LAYER2_NODE = 100
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 36
MOVING_AVERAGE_DECAY = 0.99
mnist = input_data.read_data_sets("/home/zhp/PycharmProjects/tensorflow/mnist", one_hot=True)
print mnist
N_BATCH = mnist.train.num_examples // BATCH_SIZE
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2, weights3, biases3 ):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
        layer3 = tf.nn.relu(tf.matmul(layer2, weights3) + biases3)
        return layer3
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(biases3))
        return layer3
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
    weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(x, None, weights1, biases1, weights2, biases2, weights3, biases3)
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_average.apply(tf.trainable_variables())
    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2, weights3, biases3)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variables_averages_op)
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for epoch in range(TRAINING_STEPS):
            for batch in range(N_BATCH):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            validate_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            # print sess.run(average_y)
            print("After %d training steps,test accuracy using average model is %g" % (epoch, test_acc))
            print("After %d training steps,validate accuracy using average model is %g" % (epoch, validate_acc))
        print sess.run([tf.cast(tf.argmax(average_y, 1), tf.float32),  tf.cast(tf.argmax(mnist.test.labels, 1), tf.float32)], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print sess.run([weights1, weights2, weights3])
        # saver.save(sess, "/home/zhp/PycharmProjects/tensorflow/model.ckpt")
def main(argv = None):
    train(mnist)'''
'''if __name__ == '__main__':
    tf.app.run()'''
'''rr = linee[0][0].strip().split('-')
                if rr[1] in ['01', '03', '05', '07', '08', '10', '12']:
                    DATE_NUM = 32
                elif rr[1] in ['02']:
                    DATE_NUM = 29
                else:
                    DATE_NUM = 31
                rrr=int(rr[2])
                for rrr in range(rrr,DATE_NUM):
                    qqq.append(rr[])'''
'''print a1
            print guess2
            predict_type = dataloadinput('/home/zhp/huawei 挑战赛/初赛文档/用例示例/input_5flavors_cpu_7days.txt')
            q = []
            r = []
            for i in range(3,8):
                r.append([predict_type[i][0], predict_type[i][1], predict_type[i][2]])
                q.append(predict_type[i][0])
            print predict_type
            print r
            print q
            p = []
            dic4 = {}
            for a in q:
                if a in guess2.keys():
                    p.append(guess2[a][0][1])
                if p == []:
                    p = [0]
                dic4[a] = p
                p = []
            print dic4
            Nt = q[:]
            Rh1 = []
            Rh1.append(predict_type[0])
            Rh1_cpu = int(Rh1[0][0])
            Rh1_MEM = int(Rh1[0][1])*1024
            Rh1_ying = int(Rh1[0][2])*1024
            Rh1_cpu_nochange = int(Rh1[0][0])
            Rh1_MEM_nochange = int(Rh1[0][1]) * 1024
            H = {}
            h = []
            n=1
            for v in r:
                if dic4[v[0]] == []:
                    dic4[v[0]] = [0]
                if v[0] in dic4.keys():
                    for i in range(int(dic4[v[0]][0])):
                        if int(v[1]) <= Rh1_cpu and int(v[2]) <= Rh1_MEM:
                           h.append(v[0])
                           d = len(h)
                           H[n] = h
                           Rh1_cpu = Rh1_cpu-int(v[1])
                           Rh1_MEM = Rh1_MEM-int(v[2])
                        else:
                           h = []
                           n = n+1
                           Rh_cpu = int(Rh1[0][0])
                           Rh_MEM = int(Rh1[0][1]) * 1024
                           h.append(v[0])
                           H[n] = h
                           Rh1_cpu = Rh_cpu - int(v[1])
                           Rh1_MEM = Rh_MEM - int(v[2])
            print H
            j = 0
            result = []
            file = open("result.txt", 'w')
            for flavornumber in H.keys():
                j = j+len(H[flavornumber])
            file.write(str(j)+'\n')
            for k in dic4.keys():
                result.append(k+' '+str(dic4[k][0]))
                file.write(k+' '+str(dic4[k][0])+'\n')
            file.write('\n')
            file.write(str(len(H))+'\n')
            l = []
            ll = {}
            for flavorname1 in H.keys():
                for a in q:
                  l.append([a, H[flavorname1].count(a)])
                ll[flavorname1] = l
                l=[]
            print ll
            for severce in ll.keys():
                file.write(str(severce) + ' ')
                for i in range(len(q)):
                    if int(ll[severce][i][1]) != 0:
                        file.write(str(ll[severce][i][0])+' '+str(ll[severce][i][1])+' ')
                file.write('\n')
            file.close()'''