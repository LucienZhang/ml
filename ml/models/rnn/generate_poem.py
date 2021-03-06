import sys
from gensim.models.keyedvectors import KeyedVectors
import re
from itertools import chain
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

sys.path.append('../../..')

from ml import get_model_dir, get_log_dir  # noqa
from ml.datasets import get_data_path  # noqa

wv_path = get_data_path('tencent_embedding_100k')
poem_path = get_data_path('poem')

model_name = 'rnn'
experiment_name = 'poem'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

NUM_VOCAB = 100000
EMBEDDING_DIM = 200
BATCH_SIZE = 512
EPOCHS = 200
NUM_CLEANED_POEMS = 0

wv = KeyedVectors.load(str(wv_path))
assert EMBEDDING_DIM == 200


#################################
# DATA PREPROCESS
#################################
def preprocess():
    poems = []
    with open(poem_path, "r") as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                content = content.replace('，', ',')
                if re.search(r'[_(（《\[]', content):
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poems.append(content)
            except Exception as e:
                print(line)

    # sort by length of poem
    # poems.sort(key=lambda p: len(p))
    print('number of poems:', len(poems))

    # calculate word frequency
    cnt = Counter(chain(*poems))
    print('number of characters:', sum(cnt.values()))

    oov = 0
    for word in cnt:
        if word not in wv:
            # print(word,end=' ')
            oov += cnt[word]
    print('number of oov:', oov)

    # poems=[list(jieba.cut(p)) for p in poems]
    # cnt = Counter(chain(*poems))
    # print(sum(cnt.values()))
    #
    # oov = 0
    # for word in cnt:
    #     if word not in wv:
    #         #print(word,end=' ')
    #         oov += cnt[word]
    # print(oov)

    def to_vec(poem):
        vec = []
        for w in poem:
            if w not in wv:
                return None
            vec.append(wv[w])
        return vec

    def to_sequence(poem):
        seq = []
        for w in poem:
            if w not in wv:
                return None
            seq.append(wv.vocab[w].index)
        return seq

    seq_poems = []
    for p in poems:
        seq = to_sequence(p)
        if seq:
            seq_poems.append(seq)

    print('number of cleaned poems:', len(seq_poems))
    global NUM_CLEANED_POEMS
    NUM_CLEANED_POEMS = len(seq_poems)
    # print('length:', Counter([len(p) for p in seq_poems]))

    return seq_poems


#################################
# MODEL
#################################
def train_gen(seq_poems, batch_size):
    while True:
        np.random.shuffle(seq_poems)
        for i in range(0, len(seq_poems), batch_size):
            batch = seq_poems[i:i + batch_size]
            # if len(batch) != batch_size:
            #     continue
            max_len = max([len(item) for item in batch]) + 1
            chunk = np.empty((0, max_len), dtype='int32')
            for item in batch:
                item = np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0)
                item = item[np.newaxis, :]
                chunk = np.vstack((chunk, item))
            yield chunk[:, :-1], chunk[:, 1:]


def build_model():
    embedding_weights = wv.vectors

    inputs = keras.Input(shape=(None,), dtype='int32')
    x = layers.Embedding(
        input_dim=embedding_weights.shape[0], output_dim=embedding_weights.shape[1],
        weights=[embedding_weights], trainable=False, mask_zero=True
    )(inputs)
    x = layers.LSTM(200, return_sequences=True)(x)
    # x = layers.LSTM(200, return_sequences=True)(x)
    outputs = layers.Dense(NUM_VOCAB, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            print(e)

    seq_poems = preprocess()
    gen = train_gen(seq_poems, BATCH_SIZE)
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    ckpt_cb = ModelCheckpoint(filepath=str(log_dir / '{epoch:02d}.h5'), save_weights_only=True, period=50)
    callbacks = [tb_cb, ckpt_cb]
    model.fit_generator(gen, epochs=EPOCHS, steps_per_epoch=NUM_CLEANED_POEMS // BATCH_SIZE, callbacks=callbacks)
    model.save(model_path)

    # optimizer=tf.keras.optimizers.Adam()
    #
    # @tf.function
    # def train_step(inputs, targets):
    #     with tf.GradientTape() as tape:
    #         predictions =model(inputs)
    #         loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy)


def generate(start_string):
    if start_string not in wv:
        return
    model = build_model()
    model.load_weights(str(model_path))

    num_generate = 100
    inputs = np.array(wv.vocab[start_string].index).reshape(1, 1)

    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(inputs)
        # remove the batch dimension
        predictions = tf.squeeze(predictions)
        predicted_id = tf.argmax(predictions).numpy()
        text_generated.append(wv.index2word[int(predicted_id)])

        inputs = predicted_id.reshape(1, 1)

    print(start_string + ''.join(text_generated))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == 'pre':
        preprocess()
    elif len(sys.argv) == 3 and sys.argv[1] == 'generate':
        generate(sys.argv[2])
    else:
        print('args ERROR')
