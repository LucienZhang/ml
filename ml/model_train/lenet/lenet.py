# paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning

from ml.model_train.prepare import get_model_log_dir  # noqa
from ml.model_train.datasets import get_dataset  # noqa

dataset_name = 'mnist'
model_name = 'lenet'
model_dir, log_dir = get_model_log_dir(model_name)

data, info = get_dataset(dataset_name)
data_train, data_test = data['train'], data['test']

BATCH_SIZE = 128
IMAGE_COUNT = info.splits['train'].num_examples


def preprocess(sample):
    return tf.cast(sample['image'], tf.float32) / 255.0, sample['label']


data_train = data_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
data_test = data_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# data_train = data_train.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=info.splits['train'].num_examples))
# data_train = data_train.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data_train = data_train.shuffle(buffer_size=IMAGE_COUNT)
data_train = data_train.batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data_test = data_test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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

sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks = [tb_cb]

model.fit(data_train, epochs=30, steps_per_epoch=np.ceil(IMAGE_COUNT / BATCH_SIZE), callbacks=callbacks)

model.evaluate(data_test)

model.save(model_dir / (model_name + '.h5'))
