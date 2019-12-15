# paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import os
import sys
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D, BatchNormalization, Activation, \
    Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning
sys.path.append('../../..')

from train import get_model_dir, get_log_dir  # noqa
from train.datasets import get_data_path  # noqa

dataset_name = 'fruits'
model_name = 'alex'
experiment_name = 'fruits'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

BATCH_SIZE = 2048
EPOCHS = 200

num_classes = 120

##########
# Data
##########
train_path, test_path = get_data_path('fruits')

train_gen = ImageDataGenerator(rescale=1 / 255.,
                               width_shift_range=0.125,
                               height_shift_range=0.125,
                               fill_mode='constant',
                               cval=0.,
                               horizontal_flip=True,
                               dtype='float32')
train_data = train_gen.flow_from_directory(train_path, target_size=(100, 100), class_mode='categorical',
                                           batch_size=BATCH_SIZE)

test_gen = ImageDataGenerator(rescale=1 / 255., dtype='float32')
test_data = test_gen.flow_from_directory(test_path, target_size=(100, 100), class_mode='categorical',
                                         batch_size=BATCH_SIZE)

##########
# Model
##########
inputs = keras.Input(shape=(100, 100, 3))
x = ZeroPadding2D(padding=6)(inputs)

# Stage 1
x = Conv2D(96, (4, 4), strides=(2, 2), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Stage 2
x = Conv2D(256, (5, 5), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Stage 3
x = Conv2D(384, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = Activation('relu')(x)

# Stage 4
x = Conv2D(384, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = Activation('relu')(x)

# Stage 5
x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = Flatten()(x)

# Stage 6
x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

# Stage 7
x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

# Stage 8
outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 100:
        return 0.001
    elif epoch < 150:
        return 0.0005
    else:
        return 0.0001


change_lr = LearningRateScheduler(scheduler)
tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
ckpt_cb = ModelCheckpoint(filepath=str(log_dir / '{epoch:02d}.hdf5'), monitor='val_acc', save_weights_only=True,
                          period=50)
callbacks = [change_lr, tb_cb, ckpt_cb]

model.fit_generator(train_data,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_data)

model.save(model_path)
