# paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow_datasets as tfds
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning

base_dir = Path(__file__).resolve().parents[2]
tb_path = base_dir / 'logs/lenet_mnist'
model_save_path = str(base_dir / 'weights/lenet_mnist')
data_dir = str(base_dir / 'datasets/storage')


def get_dataset(split, batch_size):
    dataset = tfds.load(name='mnist', data_dir=data_dir, shuffle_files=True, as_supervised=True, split=split)
    dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def get_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = Conv2D(6, (5, 5), activation='relu', kernel_initializer='he_normal')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(84, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(10, kernel_initializer='he_normal')(x)
    outputs = tf.nn.softmax(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train():
    BATCH_SIZE = 128
    EPOCHS = 50

    data_train = get_dataset(tfds.Split.TRAIN, BATCH_SIZE)
    data_test = get_dataset(tfds.Split.TEST, BATCH_SIZE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_writer = tf.summary.create_file_writer(str(tb_path / 'train'))
    test_writer = tf.summary.create_file_writer(str(tb_path / 'test'))
    tf.summary.trace_on(graph=True)

    model = get_model()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    @tf.function
    def train_one_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_function(y_true=y, y_pred=y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y, y_pred)

    @tf.function
    def test_one_step(x, y):
        y_pred = model(x)
        loss = loss_function(y_true=y, y_pred=y_pred)

        test_loss(loss)
        test_accuracy(y, y_pred)

    for num_epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch_num, (images, labels) in enumerate(data_train):
            train_one_step(images, labels)
            print(f'Batch {batch_num}', end='\r')
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=num_epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=num_epoch)
        print('Epoch {}, Loss: {}, Accuracy: {}'.format(num_epoch + 1, train_loss.result(),
                                                        train_accuracy.result() * 100))

        if (num_epoch + 1) % 2 == 0:
            test_loss.reset_states()
            test_accuracy.reset_states()
            for batch_num, (images, labels) in enumerate(data_test):
                test_one_step(images, labels)
                print(f'Test Batch {batch_num}', end='\r')
            with test_writer.as_default():
                tf.summary.scalar("loss", test_loss.result(), step=num_epoch)
                tf.summary.scalar("accuracy", test_accuracy.result(), step=num_epoch)
            print('Test Loss: {}, Test Accuracy: {}'.format(test_loss.result(), test_accuracy.result() * 100))
    with train_writer.as_default():
        tf.summary.trace_export(name='model_graph', step=0)
    return model


if __name__ == '__main__':
    trained_model = train()
    tf.saved_model.save(trained_model, model_save_path)
    trained_model.save(model_save_path + '.h5')  # for Netron
