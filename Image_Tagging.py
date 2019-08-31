
import numpy as np
import tensorflow as tf

import vgg_16.VGG16 as vgg

from Define import *

init_fn = tf.contrib.layers.xavier_initializer()

def Global_Average_Pooling(x, stride=1):
    return tf.layers.average_pooling2d(inputs=x, pool_size=np.shape(x)[1:3], strides=stride)

def Image_Tagging(x, is_training, classes, decision_model = False):
    predict_dic = {}
    x = x - [103.94, 116.78, 123.68]
    
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(x, num_classes = 1000, is_training = is_training, dropout_keep_prob = 0.5)

    predict_dic['LSEP'] = end_points['vgg_16/fc8']
    
    if decision_model:
        with tf.variable_scope('Label_Decision'):
            # Confidence
            x = tf.layers.dense(predict_dic['LSEP'], classes, kernel_initializer = init_fn)
            x = tf.layers.batch_normalization(x, training = is_training)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate = 0.5, training = is_training)
            
            conf_logits = tf.layers.dense(x, classes, kernel_initializer = init_fn)
            predict_dic['Confidence'] = tf.nn.sigmoid(conf_logits, name = 'conf/sigmoid')

            # TopK
            x = tf.layers.dense(predict_dic['LSEP'], 100, kernel_initializer = init_fn)
            x = tf.layers.batch_normalization(x, training = is_training)
            x = tf.nn.relu(x)

            topk_logits = tf.layers.dense(x, MAX_TOP_K, kernel_initializer = init_fn)
            predict_dic['TopK'] = tf.nn.softmax(topk_logits, axis = -1, name = 'topk/softmax')
    
    return predict_dic

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 224, 224, 3], name = 'images')
    
    predict_dic = Image_Tagging(input_var, False, 1000, decision_model = True)
    print(predict_dic)

