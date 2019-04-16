import numpy as np
import tensorflow as tf

from Define import *

def BatchNormalization(x, is_training, scope):

    #return tf.layers.batch_normalization(inputs = x, training = training, name = scope, reuse = reuse),

    with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(is_training,
                       lambda : tf.contrib.layers.batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda : tf.contrib.layers.batch_norm(inputs=x, is_training=is_training, reuse=True))

def conv_bn_relu(x, kernel_size, num_filters, is_training, layer_name):
    init = tf.contrib.layers.xavier_initializer()

    x = tf.layers.conv2d(inputs = x, filters = num_filters, kernel_size = kernel_size, kernel_initializer = init, strides = 1, padding = 'SAME', name = layer_name)
    x = BatchNormalization(x, is_training, layer_name + '_bn')
    x = tf.nn.relu(x)

    return x

def passthrough_layer(a, b, kernel, depth, size, is_training, name):
	
    b = conv_bn_relu(b, kernel, depth, is_training, name)
    b = tf.space_to_depth(b, size)
    y = tf.concat([a, b], axis=3)
	
    return y

def YOLOv2(x, is_training):
    x = x / 127.5 - 1

    x = conv_bn_relu(x, (3, 3), 32, is_training, 'conv1')
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool1')
    x = conv_bn_relu(x, (3, 3), 64, is_training, 'conv2')
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool2')
    
    x = conv_bn_relu(x, (3, 3), 128, is_training, 'conv3')
    x = conv_bn_relu(x, (1, 1), 64, is_training, 'conv4')
    x = conv_bn_relu(x, (3, 3), 128, is_training, 'conv5')
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool5')

    x = conv_bn_relu(x, (3, 3), 256, is_training, 'conv6')
    x = conv_bn_relu(x, (1, 1), 128, is_training, 'conv7')
    x = conv_bn_relu(x, (3, 3), 256, is_training, 'conv8')
    x = tf.layers.max_pooling2d(inputs = x, pool_size = [2, 2], strides = 2, name = 'maxpool8')

    x = conv_bn_relu(x, (3, 3), 512, is_training, 'conv9')
    x = conv_bn_relu(x, (1, 1), 256, is_training, 'conv10')
    x = conv_bn_relu(x, (3, 3), 512, is_training, 'conv11')
    x = conv_bn_relu(x, (1, 1), 256, is_training, 'conv12')
    passthrough = conv_bn_relu(x, (3, 3), 512, is_training, 'conv13')
    x = tf.layers.max_pooling2d(inputs = passthrough, pool_size = [2, 2], strides = 2, name = 'maxpool13')
    
    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv14')
    x = conv_bn_relu(x, (1, 1), 512, is_training, 'conv15')
    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv16')
    x = conv_bn_relu(x, (1, 1), 512, is_training, 'conv17')
    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv18')

    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv19')
    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv20')
    x = passthrough_layer(x, passthrough, (3, 3), 64, 2, is_training, 'conv21')					 
    x = conv_bn_relu(x, (3, 3), 1024, is_training, 'conv22')
    x = conv_bn_relu(x, (1, 1), N_ANCHORS * (CLASSES + BOX_SIZE), is_training, 'conv23')

    x = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, CLASSES + BOX_SIZE), name='outputs')					
    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [10, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    training_var = tf.placeholder(tf.bool)
    yolov2 = YOLOv2(input_var, training_var)
    print(yolov2)
