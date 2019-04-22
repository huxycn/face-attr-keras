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
  @File        : test.py
  @Author      : huxiaoyang :)
  @Created on  : 2019-03-19-3-30 18:39
  @Description : 
            
"""


import os
import numpy as np
from PIL import Image

from models.lenet5 import lenet5_model

from config import *

model = lenet5_model()
# model.load_weights('../weights/lenet5_weights_produce.h5')

def eyeglasses_predict(img_pil):
    img_pil = img_pil.resize((100, 100))
    img_array = np.asarray(img_pil)
    img_array_ = img_array / 255
    x = img_array_.reshape((1, ) + img_array.shape)
    result = model.predict(x)
    p_no = result[0][0]
    p_yes = result[0][1]
    if p_yes > p_no:
        return True
    else:
        return False




if __name__ == '__main__':

    nb_correct = 0
    nb_total = 0

    img_dir = os.path.join(DATA_DIR, 'real_world_img/glasses')
    for img in os.listdir(img_dir):
        nb_total += 1
        img_pil = Image.open(os.path.join(img_dir, img))

        if eyeglasses_predict(img_pil):
            nb_correct += 1
    img_dir = os.path.join(DATA_DIR, 'real_world_img/noglasses')
    for img in os.listdir(img_dir):
        nb_total += 1
        img_pil = Image.open(os.path.join(img_dir, img))
        if not eyeglasses_predict(img_pil):
            nb_correct += 1

    print('ACC:', 1. * nb_correct / nb_total)
