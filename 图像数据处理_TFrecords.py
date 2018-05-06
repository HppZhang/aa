# -*-coding: utf-8-*-
import tensorflow as tf
#将MNIST数据集所有训练数据存储到一个TFrecord文件中
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
mnist = input_data.read_data_sets("/home/zhp/PycharmProjects/tensorflow/mnist", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_example = mnist.train.num_examples
filename = '/home/zhp/PycharmProjects/tensorflow/output.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_example):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={'pixels': _int64_feature(pixels),
                                                                   'label': _int64_feature(np.argmax(labels[index])),
                                                                   'image_raw': _bytes_feature(image_raw)}))
writer.write(example.SerializeToString())
writer.close()
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['/home/zhp/PycharmProjects/tensorflow/output.tfrecords'])
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image_raw': tf.FixedLenFeature([], tf.string),
                                       'pixels': tf.FixedLenFeature([], tf.int64),
                                       'label': tf.FixedLenFeature([], tf.int64),
                                   })
images = tf.decode_raw(features['image_raw'], tf.uint8)
pixels = tf.cast(features['pixels'], tf.int32)
labels = tf.cast(features['label'], tf.int32)
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])