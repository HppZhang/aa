# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
LAYER2_NODE = 100
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 300
MOVING_AVERAGE_DECAY = 0.99
mnist = input_data.read_data_sets("/home/zhp/PycharmProjects/tensorflow/mnist", one_hot=True)
N_BATCH = mnist.train.num_examples // BATCH_SIZE
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2, weights3, biases3 ):
    if avg_class == None:
        with tf.name_scope("layer1"):
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        with tf.name_scope("layer2"):
            layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
        with tf.name_scope("layer3"):
            layer3 = tf.nn.relu(tf.matmul(layer2, weights3) + biases3)
        return layer3
    else:
        with tf.name_scope("layer1"):
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        with tf.name_scope("layer2"):
            layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        with tf.name_scope("layer3"):
            layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(biases3))
        return layer3
def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    with tf.name_scope("layer1"):
        weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
        biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    with tf.name_scope("layer2"):
        weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
        biases2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
    with tf.name_scope("layer3"):
        weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
        biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("moving_average"):
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_average.apply(tf.trainable_variables())
    with tf.name_scope("output"):
        y = inference(x, None, weights1, biases1, weights2, biases2, weights3, biases3)
        average_y = inference(x, variable_average, weights1, biases1, weights2, biases2, weights3, biases3)
    with tf.name_scope("loss_founction"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
        loss = cross_entropy_mean + regularization
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.group(train_step, variables_averages_op)
    with tf.name_scope("accurent"):
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    writer = tf.summary.FileWriter("/home/zhp/PycharmProjects/tensorflow/summary", tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        print mnist.test.labels[0]
        print sess.run(tf.argmax(y_), feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print sess.run(
            y[0],
            feed_dict={x: mnist.test.images, y_: mnist.test.labels})
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
    train(mnist)
if __name__ == '__main__':
    tf.app.run()