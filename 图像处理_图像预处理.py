# -*-coding: utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)
def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.Constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distort_image = tf.slice(image, bbox_begin, bbox_size)
    #将随机截图的图像调整为神经网络输入层的大小。大小调整的算法是随机的
    distort_image = tf.image.resize_images(
        distort_image, [height, width], method=np.random.randint(4)
    )
    # 随机左右翻转图像
    distort_image = tf.image.random_flip_left_right(distort_image)
    # 使用一种随机的顺序调整图像色彩
    distort_image = distort_color(distort_image, np.random.randint(1))
    return distort_image

with tf.Session() as Sess:
    image_raw_data = tf.gfile.FastGFile("/home/zhp/PycharmProjects/cat.jpg", 'r').read()
    ima_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    b = np.array([2, 4, 5])
    e = np.array([2, 4, 5])
    c = np.matrix([[1, 2, 3],
                   [2, 3, 4],
                   [3, 4, 5]])
    d = c**(-1)    # c的逆矩阵
    print d, b+e
    print b.shape, b.size, c.size, c.shape
    # 运行6次获得6中不同的图像，在图中显示效果
    for i in range(6):
        # 将图像的尺寸调整为299*299
        result = preprocess_for_train(ima_data, 299, 299, boxes)

        plt.imshow(result.eval())
        plt.show()

