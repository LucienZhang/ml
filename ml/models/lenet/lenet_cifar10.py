# paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning
sys.path.append('../../..')

from ml import get_model_dir, get_log_dir  # noqa
from ml.datasets import get_dataset  # noqa

dataset_name = 'cifar10'
model_name = 'lenet'
experiment_name = 'cifar10_rewrite_test'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

BATCH_SIZE = 128
EPOCHS = 200

data, info = get_dataset(dataset_name)
NUM_CLASS = info.features['label'].num_classes
NUM_TRAIN = info.splits['train'].num_examples
NUM_VAL = info.splits['test'].num_examples

##########
# Data
##########

data_train, data_test = data['train'], data['test']
data_train = data_train.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
data_test = data_test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def normalization(images):
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    images = images.astype('float32')
    images = (images - mean) / std
    return images


def augmentation(image):
    image = keras.preprocessing.image.random_shift(image, 0.125, 0.125, row_axis=0, col_axis=1, channel_axis=2,
                                                   fill_mode='constant', cval=0.0)
    image = tf.image.random_flip_left_right(image)
    return image


def train_gen(dataset, num_class):
    while True:
        for images, labels in tfds.as_numpy(dataset):
            labels = keras.utils.to_categorical(labels, num_class)
            images = normalization(images)
            for i in range(len(images)):
                images[i] = augmentation(images[i])
            yield images, labels


def test_gen(dataset, num_class):
    while True:
        for images, labels in tfds.as_numpy(dataset):
            labels = keras.utils.to_categorical(labels, num_class)
            images = normalization(images)
            yield images, labels


##########
# Model
##########
inputs = keras.Input(shape=(32, 32, 3))
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

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    elif epoch < 150:
        return 0.005
    else:
        return 0.001


change_lr = LearningRateScheduler(scheduler)
tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks = [change_lr, tb_cb]

model.fit_generator(train_gen(data_train, NUM_CLASS),
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_gen(data_test, NUM_CLASS),
                    steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
                    validation_steps=NUM_VAL // BATCH_SIZE,
                    )

model.save(model_path)
