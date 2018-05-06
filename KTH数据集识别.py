# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse as args
import matplotlib.pyplot as plt
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.inception_v3_matt import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*"))) #dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
  return cnt  # 返回类别数目

# get_nb_files('/home/ubuntu/keras/animal5/train')
# 数据准备
IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3指定的图片尺寸
FC_SIZE = 1024                # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量


train_dir = '/home/zhp/PycharmProjects/human_kth/train'  # 训练集数据
val_dir = '/home/zhp/PycharmProjects/human_kth/test' # 验证集数据
# nb_classes = 6
nb_epoch = 500
batch_size = 10

nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(nb_epoch)                # epoch数量
batch_size = int(batch_size)           

#　图片生成器
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(val_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size, class_mode='categorical')
# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x) # 输入shape:若从未设置过，则为“channels_last”,‘channels_last’模式下，
                                    # 为形如（samples，rows, cols，channels）的4D张量
                                    #输出shape:形如(nb_samples, channels)的2D张量
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init#输入
#形如(batch_size, ..., input_dim)的nD张量，最常见的情况为(batch_size, input_dim)的2D张量
#输出
#形如(batch_size, ..., units)的nD张量，最常见的情况为(batch_size, units)的2D张量
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=SGD(lr=0.0005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 定义网络框架
base_model = InceptionV3(weights='imagenet', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层
setup_to_transfer_learn(model, base_model)                      # 冻结base_model所有层
# 模式一训练
history_tl = model.fit_generator(train_generator, nb_epoch=nb_epoch, steps_per_epoch=1, validation_data=validation_generator, validation_steps=1, class_weight='auto')
def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# 设置网络结构
setup_to_finetune(model)
# 模式二训练
print "模式2"
history_ft = model.fit_generator(train_generator, steps_per_epoch=1, nb_epoch=nb_epoch, validation_data=validation_generator, validation_steps=1, class_weight='auto')
# 模型保存
# model.save(args.output_model_file)
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r-')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

# 训练的acc_loss图
plot_training(history_tl)
plot_training(history_ft)
