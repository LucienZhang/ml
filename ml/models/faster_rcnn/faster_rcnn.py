from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def rpn(feature_map, anchors_per_location=9):
    shared = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)
    # Anchor class (foreground, background)
    # [batch, height, width, anchors_per_location * 2]
    x = layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)
    # Reshape to [batch, anchors, 2]
    x = layers.Reshape((-1, 2))(x)
    rpn_class = layers.Activation('softmax', name='rpn_class_probs')(x)

    # Bounding box refinement
    # [batch, height, width, anchors_per_location * (x, y, log(w), log(h))]
    x = layers.Conv2D(4 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 4]
    rpn_bbox = layers.Reshape((-1, 4))(x)

    return rpn_class, rpn_bbox

def get_anchors():
    anchors=[]
    scales=(8,16,32)
    ratios=(0.5,1,2)
    # for

def build_model():
    inputs = layers.Input(shape=(None, None, 3))  # default shape is 224*224*3
    x = preprocess_input(inputs)
    backbone = VGG16(weights='imagenet', include_top=False)
    feature_map = backbone(x)
    rpn_class, rpn_bbox = rpn(feature_map)
    anchors=get_anchors()
