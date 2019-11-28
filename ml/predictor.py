from pathlib import Path

from PIL import Image
import numpy as np
from tensorflow import keras

model_dir = Path(__file__).resolve().parent / 'model_train/models'
model_names = ('lenet',)
models = {}
implementors = {'mnist': 'lenet'}


def init_predictor():
    for name in model_names:
        models[name] = keras.models.load_model(model_dir / (name + '.h5'))


def predict(task, inputs):
    model_name = implementors[task]
    model = models[model_name]
    if model_name == 'lenet':
        images = np.empty((0, 28, 28, 1), dtype='float32')
        for sample in inputs:
            img = Image.open(sample).resize((28, 28)).convert('L')
            # img.save('1.png')
            img = np.array(img, dtype='float32')
            img = img.reshape((-1, 28, 28, 1))
            img /= 255.0
            images = np.vstack((images, img))
        prob_vec = model.predict(images)
        results = np.argmax(prob_vec, axis=1)
        probabilities = np.max(prob_vec, axis=1)
    else:
        raise NameError(f'model name {model_name} not found!')
    return results, probabilities
