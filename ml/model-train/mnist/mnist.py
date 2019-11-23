# -*- coding: utf-8 -*-
# @Author: Lucien Zhang
# @Date:   2019-09-08 16:48:20
# @Last Modified by:   Lucien Zhang
# @Last Modified time: 2019-09-08 17:12:08
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

mdl_path='./model/mnist.h5'
tf.debugging.set_log_device_placement(True)

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(32,3,activation='relu',input_shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_train = x_train.reshape(-1,28, 28, 1)
x_test = x_test.reshape(-1,28, 28, 1)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

model.save(mdl_path)
