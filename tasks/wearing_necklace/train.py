#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#       ______                       _       __    __     ___   ____ _______     ______            ____             
#      / ____/___  ____  __  _______(_)___ _/ /_  / /_   |__ \ / __ <  / __ \   / ____/___  ____ _/ __ \____ ___  __
#     / /   / __ \/ __ \/ / / / ___/ / __ `/ __ \/ __/   __/ // / / / / /_/ /  / / __/ __ \/ __ `/ /_/ / __ `/ / / /
#    / /___/ /_/ / /_/ / /_/ / /  / / /_/ / / / / /_    / __// /_/ / /\__, /  / /_/ / /_/ / /_/ / ____/ /_/ / /_/ / 
#    \____/\____/ .___/\__, /_/  /_/\__, /_/ /_/\__/   /____/\____/_//____/   \____/\____/\__, /_/    \__,_/\__, /  
#              /_/    /____/       /____/                                                /____/            /____/   
#
# ======================================================================================================================

"""
  @File        : train.py
  @Author      : huxiaoyang :)
  @Created on  : 2019-03-19-3-28 22:32
  @Description : 
            
"""


import datetime
import keras
from keras import models, layers, regularizers
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import densenet

from config import DATA_DIR
from dataset.celeba import CelebA
from models.alexnet import alexnet_model
from models.lenet5_with_dropout import lenet5_model


# 模型超参数
H = 100    # 图像高/行数
W = 100    # 图像宽/列数
C = 3      # 图像通道数/深度
batch_size = 32
n_epoch = 300

# tensor board 回调
tb_callback = TensorBoard(log_dir='/home/huxiaoyang/PycharmProjects/face_attr/tasks/wearing_necklace/tb_logs',
                          histogram_freq=0,
                          batch_size=batch_size,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None)


def main():
    # 加载数据集
    celeba = CelebA(attr='Wearing_Earrings')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = celeba.load_data()

    # 数据集预处理
    y_train = keras.utils.to_categorical(y_train, 2)
    y_val = keras.utils.to_categorical(y_val, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    data_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    data_gen.fit(X_train)

    model = alexnet_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    # 模型训练
    model.fit_generator(data_gen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=n_epoch,
                        validation_data=(X_val, y_val),
                        callbacks=[tb_callback])
    # 保存训练过程和模型参数
    cur_time = datetime.datetime.now().strftime('%Y%m%d_h%Hm%M')
    model.save_weights('weights.{}.h5'.format(cur_time))

    # 测试集准确率
    result = model.evaluate(x=X_test, y=y_test, batch_size=batch_size)
    print(result)

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    seconds = time.time() - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print('use time: {}h {}m {}s'.format(h, m, s))
