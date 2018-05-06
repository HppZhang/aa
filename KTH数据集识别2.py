# -*- coding: utf-8 -*-
# 载入与模型网络构建
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(299, 299,3)))
# filter大小3*3，数量32个，原始图像大小3,150,150
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))   #                               matt,几个分类就要有几个dense
model.add(Activation('softmax'))#                     matt,多分类
model.compile(loss='categorical_crossentropy',                                 # matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/wyb/tf-notebooks/data/KTH-OUT/training', 
        target_size=(299, 299),  # all images will be resized to 150x150
        batch_size=40,
        class_mode='categorical')                               # matt，多分类

validation_generator = test_datagen.flow_from_directory(
        '/home/wyb/tf-notebooks/data/KTH-OUT/test',
        target_size=(299, 299),
        batch_size=40,
        class_mode='categorical')                             # matt，多分类
model.fit_generator(
        train_generator,
        steps_per_epoch=225,
        nb_epoch=50,
        validation_data=validation_generator,
        validation_steps=20)
# samples_per_epoch，相当于每个epoch数据量峰值，每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
model.save_weights('/home/wyb/tf-notebooks/saved_model/first_try_KTH.h5') 
