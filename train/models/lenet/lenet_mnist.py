# paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning
sys.path.append('..')

from train import get_model_dir, get_log_dir  # noqa
from train.datasets import get_dataset  # noqa

dataset_name = 'mnist'
model_name = 'lenet'
experiment_name = 'mnist1'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

BATCH_SIZE = 128
EPOCHS = 50

data, info = get_dataset(dataset_name)
data_train, data_test = data['train'], data['test']


def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


data_train = data_train.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
data_train = data_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

data_test = data_test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
data_test = data_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Model
inputs = keras.Input(shape=(28, 28, 1))
x = Conv2D(6, (5, 5), activation='relu', kernel_initializer='he_normal')(inputs)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(120, activation='relu', kernel_initializer='he_normal')(x)
x = Dense(84, activation='relu', kernel_initializer='he_normal')(x)
outputs = Dense(10, activation='softmax', kernel_initializer='he_normal')(x)

model = Model(inputs=inputs, outputs=outputs)

sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks = [tb_cb]

model.fit(data_train, epochs=EPOCHS, callbacks=callbacks, validation_data=data_test)

# model.save(model_path)
