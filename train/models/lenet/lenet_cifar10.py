# paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning
sys.path.append('../../..')

from train import get_model_dir, get_log_dir  # noqa

dataset_name = 'cifar10'
model_name = 'lenet'
experiment_name = 'cifar10_my'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

BATCH_SIZE = 128
EPOCHS = 200

num_classes = 10

##########
# Data
##########

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

train_gen = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True,
                               width_shift_range=0.125,
                               height_shift_range=0.125,
                               fill_mode='constant',
                               cval=0.,
                               horizontal_flip=True)
train_gen.fit(x_train)

test_gen = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True)
test_gen.fit(x_train)

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

model.fit_generator(train_gen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_gen.flow(x_test, y_test, batch_size=BATCH_SIZE))

# model.fit(data_train, epochs=EPOCHS, callbacks=callbacks, validation_data=data_test)

model.save(model_path)
