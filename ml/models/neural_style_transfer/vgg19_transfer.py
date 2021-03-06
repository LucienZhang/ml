import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
import time
from functools import partial
import numpy as np
import PIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to suppress AVX2 warning
sys.path.append('../../..')

from ml import get_image_dir  # noqa

model_name = 'vgg19_style_trans'
experiment_name = 'eagle'
image_dir = get_image_dir() / experiment_name


#############################
# Utils
#############################
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


#############################
# Model
#############################
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        style_layer_names = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        content_layer_names = ['block5_conv2']

        wanted_layer_names = style_layer_names + content_layer_names
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        wanted_layers = [vgg.get_layer(name).output for name in wanted_layer_names]
        self.backbone = tf.keras.Model(vgg.input, wanted_layers)
        self.backbone.trainable = False
        self.style_layer_names = style_layer_names
        self.content_layer_names = content_layer_names
        self.num_style_layers = len(style_layer_names)
        self.num_content_layers = len(content_layer_names)

    def call(self, input_image):
        """Expects float input in [0,1]"""
        input_image = input_image * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input_image)
        wanted_layers_outputs = self.backbone(preprocessed_input)
        style_outputs, content_outputs = (wanted_layers_outputs[:self.num_style_layers],
                                          wanted_layers_outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layer_names, style_outputs)}

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layer_names, content_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs, style_weight, style_targets, num_style_layers,
                       content_weight, content_targets, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def get_train_step(extractor, loss_func, optimizer):
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = loss_func(outputs)

            grad = tape.gradient(loss, image)
            optimizer.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))

    return train_step


def main():
    content_image_path = image_dir / 'content.jpg'
    style_image_path = image_dir / 'style.jpg'
    content_image = load_img(str(content_image_path))
    style_image = load_img(str(style_image_path))

    extractor = StyleContentModel()
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    ##############
    # Hyper parameters
    ##############
    STYLE_WEIGHT = 1e-2
    CONTENT_WEIGHT = 1e4
    EPOCHS = 10
    STEPS_PER_EPOCH = 100

    loss_func = partial(style_content_loss, style_weight=STYLE_WEIGHT, style_targets=style_targets,
                        num_style_layers=extractor.num_style_layers,
                        content_weight=CONTENT_WEIGHT, content_targets=content_targets,
                        num_content_layers=extractor.num_content_layers)
    train_step = get_train_step(extractor, loss_func, optimizer)
    ##############
    # Training
    ##############
    print('Training')
    start = time.time()
    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            train_step(image)
            print(step, end='\r', flush=True)
        print(f"Train step: {(epoch + 1) * STEPS_PER_EPOCH}")
        # tensor_to_image(image).save(f"{epoch}.jpg")
    tensor_to_image(image).save(str(image_dir / "result.jpg"))
    end = time.time()
    print("Total time: {:.1f}".format(end - start))


if __name__ == '__main__':
    main()
