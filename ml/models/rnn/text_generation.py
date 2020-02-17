import sys
import jieba
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
import pickle
from pathlib import Path
from itertools import chain

sys.path.append('../../..')

from ml import get_model_dir, get_log_dir  # noqa
from ml.datasets import get_data_path  # noqa

data_path = get_data_path('shuihu')

model_name = 'rnn'
experiment_name = 'shuihu'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)

model_file_name = model_name
if experiment_name:
    model_file_name += '_' + experiment_name
model_file_name += '.h5'
model_path = model_dir / model_file_name

NUM_VOCAB = 56131
EMBEDDING_DIM = 256
SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 30
NUM_CHARS = 560761

tokenizer_path = model_dir / 'tokenizer.pickle'


def get_lines():
    with open(data_path) as f:
        lines = f.readlines()
    lines = [line.encode('utf-8').decode('utf-8-sig').replace(u'\u3000', u' ').replace(u'\xa0', u' ').strip() for line
             in lines]
    lines = [line for line in lines if line]
    lines = [list(jieba.cut(line)) for line in lines]
    return lines


def get_tokenizer():
    if tokenizer_path.exists():
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        lines = get_lines()
        tokenizer = Tokenizer(num_words=NUM_VOCAB)
        tokenizer.fit_on_texts(lines)
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=NUM_VOCAB, output_dim=EMBEDDING_DIM, batch_input_shape=(BATCH_SIZE, None)))
    model.add(layers.GRU(1024,
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(NUM_VOCAB))
    return model


def train():
    tokenizer = get_tokenizer()
    lines = get_lines()
    lines = tokenizer.texts_to_sequences(lines)
    chars = list(chain(*lines))
    char_dataset = tf.data.Dataset.from_tensor_slices(chars)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    model = build_model()
    model.load_weights(log_dir)

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(log_dir / 'ckpt_{epoch}'),
        save_weights_only=True,
        period=5)

    model.fit(dataset, epochs=EPOCHS, steps_per_epoch=100, callbacks=[checkpoint_callback])


def generate_text(model, start_string):
    num_generate = 1000
    tokenizer = get_tokenizer()
    inputs = tokenizer.texts_to_sequences([' '.join(jieba.cut(start_string))])

    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(inputs)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        inputs = tf.expand_dims([predicted_id], 0)

        text_generated.append(tokenizer.sequences_to_texts([predicted_id])[0])

    return start_string + ''.join(text_generated)


def generate():
    model = tf.keras.models.load_model(model_path)
    model.build(tf.TensorShape([1, None]))
    print(generate_text(model, start_string=u"昨天"))


if __name__ == '__main__':
    train()
