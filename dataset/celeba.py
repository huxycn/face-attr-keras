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
  @File        : celeba.py
  @Author      : huxiaoyang :)
  @Created on  : 2019-03-19-3-28 22:25
  @Description : 
            
"""


import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array

from config import *


class CelebA():
    def __init__(self, data_root=DATA_DIR, resize_shape=(100, 100), attr='', train_val_test_split=(.8, .1, .1),
                 data_balance_rate=0):
        """

        :param data_root: 数据目录，包括图片和标签
        :param resize_shape: 图片缩放后尺寸，模型输入的尺寸
        :param attr: 人脸属性，眼镜、帽子、耳环、项链、模糊、黑发
        :param data_balance_rate: 数据平衡度，为0时数据正负例1:1，为1时保持原始比例
        """
        self.data_root = data_root
        self.resize_shape = resize_shape
        self.attr = attr
        self.img_dir = os.path.join(data_root, 'img_align_celeba')
        self.attr_file = os.path.join(data_root, 'list_attr_celeba.txt')
        attr_df = pd.read_csv(self.attr_file, delim_whitespace=True, header=1)[[attr]]

        # 参数data_balance_rate调节数据集平衡度，为0时1：1，为1时保持原始比例
        pos_df = attr_df[attr_df[attr] == 1]
        neg_df = attr_df[attr_df[attr] == -1]
        nb_pos = len(pos_df)
        nb_neg = len(neg_df)
        if nb_pos > nb_neg:
            pos_df = pos_df[:int(data_balance_rate * nb_pos + (1-data_balance_rate) * nb_neg)]
        else:
            neg_df = neg_df[:int(data_balance_rate * nb_neg + (1-data_balance_rate) * nb_pos)]
        attr_df = pd.concat([pos_df, neg_df])
        attr_df = attr_df.sample(frac=1.0)    # shuffle

        # 按train_val_test_split给出的比例划分训练集、验证集、测试集
        offset1 = int(len(attr_df) * train_val_test_split[0])
        offset2 = int(len(attr_df) * (train_val_test_split[0] + train_val_test_split[1]))
        self.train_df = attr_df[:offset1]
        self.val_df = attr_df[offset1: offset2]
        self.test_df = attr_df[offset2:]

    def load_data(self):
        """
            加载数据集
        :return: (X_train, y_train), (X_val, y_val), (X_test, y_test) 均为ndarray类型
        """
        # 加载训练集
        def read_data(split=None):
            print('loading {} set ...'.format(split))
            imgs = None
            labels = None
            if split == 'train':
                imgs = self.train_df.index.values
                labels = self.train_df.values
            elif split == 'val':
                imgs = self.val_df.index.values
                labels = self.val_df.values
            elif split == 'test':
                imgs = self.test_df.index.values
                labels = self.test_df.values
            X = []
            y = []
            start = time.time()
            for i in tqdm(range(len(imgs))):
                img = load_img(os.path.join(self.img_dir, imgs[i]), target_size=(self.resize_shape))
                x = img_to_array(img)
                X.append(x)
                if labels[i] == -1:
                    y.append(0)
                else:
                    y.append(1)
            print('use time: {}'.format(time.time() - start))
            return (np.asarray(X), np.asarray(y))

        X_train, y_train = read_data('train')
        X_val, y_val = read_data('val')
        X_test, y_test = read_data('test')

        print('X_train: {}    y_train: {}'.format(X_train.shape, y_train.shape))
        print('X_val: {}    y_val: {}'.format(X_val.shape, y_val.shape))
        print('X_test: {}    y_test: {}'.format(X_test.shape, y_test.shape))

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
