from keras_applications import vgg19
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import scipy.io

# configure for performance!
# src: https://www.freecodecamp.org/news/how-a-badly-configured-tensorflow-in-docker-can-be-10x-slower-than-expected-3ac89f33d625/
config = tf.ConfigProto()
# set parallelism thread/nodes
config.inter_op_parallelism_threads = 11
config.intra_op_parallelism_threads = 1
# gpu growth
#config.gpu_options.allow_growth = True
# set config
# set_session(tf.Session(config=config))
session = tf.Session(config=config)

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def load_vgg(mat_path):
    mat = scipy.io.loadmat(mat_path)
    mean = mat['meta']['normalization'][0][0][0][0][2][0][0]
    weights = mat['layers'][0]
    return weights, mean


def preloaded_vgg(weights, content_img, pooling):
    net = {}
    layer = content_img
    for i, name in enumerate(VGG19_LAYERS):
        layer_type = name[:4]
        if (layer_type == 'conv'):
            kernels, bias = weights[i][0][0][2][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            # create conv layers
            conv = tf.nn.conv2d(layer, tf.constant(weights),
                                strides=(1, 1, 1, 1), padding="SAME")
            layer = tf.nn.bias_add(conv, bias)
        elif layer_type == 'relu':
            layer = tf.nn.relu(layer)
        elif layer_type == 'pool':
            # use average or max pooling from config
            if pooling == 'avg':
                layer = tf.nn.avg_pool(layer, ksize=(
                    1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
            else:
                layer = tf.nn.max_pool(layer, ksize=(
                    1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        # add layer
        net[name] = layer
    assert len(net) == len(VGG19_LAYERS)
    return net


def preprocess(img, mean):
    return img - mean


def unprocess(img, mean):
    return img + mean
