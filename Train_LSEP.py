
import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Utils import *
from Define import *
from Image_Tagging import *
from NUS_WIDE_Helper import *
from Teacher import *

from loss.LSEP_Loss import *
from loss.TopK_Loss import *
from loss.Focal_Loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 1. dataset
helper = NUS_WIDE_Helper(LABEL_TXT_PATH)

train_threads =[]
for i in range(5):
    train_thread = Teacher('./dataset/_train.txt', helper, max_data_size = 20)
    train_threads.append(train_thread)

valid_json_paths = read_txt('./dataset/_valid.txt', REPLACE_JSON_DIR)

open('./log.txt', 'w').close()
log_print('[i] Train : {}'.format(len(train_thread.json_paths)))
log_print('[i] Valid : {}'.format(len(valid_json_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_conf_var = tf.placeholder(tf.float32, [None, CLASSES])
is_training = tf.placeholder(tf.bool)

print('[i] generate graph !')
predict_dic = Image_Tagging(input_var, is_training, CLASSES, decision_model = False)

vars = tf.trainable_variables()

lsep_loss_op = Log_Sum_Exp_Pairwise_Loss(predict_dic['LSEP'], label_conf_var, BATCH_SIZE)
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY

loss_op = lsep_loss_op + l2_reg_loss_op

tf.summary.scalar('Loss', loss_op)
tf.summary.scalar('LSEP_Loss', lsep_loss_op)
tf.summary.scalar('L2_Regularization_Loss', l2_reg_loss_op)
summary_op = tf.summary.merge_all()
print('[i] set LSEP/TopK/Confidense loss op !')

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    print('[i] ready Optimizer', end = '')
    train_op = tf.train.MomentumOptimizer(learning_rate_var, 0.9).minimize(loss_op)
    print('\r[i] set Optimizer')

# 3. train
print('[i] ready tf.Session()', end = '')
sess = tf.Session() 
print('\r[i] set tf.Session()       ')

print('[i] ready Session !', end = '')
sess.run(tf.global_variables_initializer())
print('\r[i] set global_variables_initializer !')

'''
pretrained_vars = []
for var in tf.trainable_variables():
    if 'Inception' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './inception_resnet_v2_model/inception_resnet_v2_2016_08_30.ckpt')
log_print('[i] restored imagenet model (inception resnet v2)')
'''

saver = tf.train.Saver()
saver.restore(sess, './model/Inception_ResNet_v2_30000.ckpt')

print('[i] ready thread')
for index, train_thread in enumerate(train_threads):
    print('[i] thread.start() {}'.format(index))
    train_thread.start()
print('[i] start all thread')

max_epoch = 50
learning_rate = INIT_LEARNING_RATE
max_iteration = len(train_threads[0].json_paths) // BATCH_SIZE * max_epoch
decay_iteration = np.asarray([0.5 * max_iteration, 0.75 * max_iteration], dtype = np.int32)

best_valid_lsep_loss = 999.0
valid_iteration = len(valid_json_paths) // BATCH_SIZE

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
lsep_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train_LSEP')

for iter in range(1, max_iteration):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))
        
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_label_topk_data, batch_label_conf_data = train_thread.get_batch_data()        
                break
    
    _feed_dict = {input_var : batch_image_data, label_conf_var : batch_label_conf_data, is_training : True, learning_rate_var : learning_rate}
    _, loss, lsep_loss, l2_reg_loss, summary = sess.run([train_op, loss_op, lsep_loss_op, l2_reg_loss_op, summary_op], feed_dict = _feed_dict)
    
    if np.isnan(loss):
        print('[!]', loss, lsep_loss, l2_reg_loss)
        input()
    
    loss_list.append(loss)
    lsep_loss_list.append(lsep_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    train_writer.add_summary(summary, iter)
    
    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        lsep_loss = np.mean(lsep_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)

        log_print('[i] iter : {}, loss : {:.4f}, lsep_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, lsep_loss, l2_reg_loss, train_time))

        loss_list = []
        lsep_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        valid_lsep_loss = 0.0
        batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
        batch_label_conf_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)

        for i in range(valid_iteration):
            sys.stdout.write('\r[{}/{}]'.format(i + 1, valid_iteration))
            sys.stdout.flush()

            batch_json_paths = valid_json_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            for index, json_path in enumerate(batch_json_paths):
                image_data, label_data = helper.get_test_data(json_path, get_tags = False)

                batch_image_data[index] = image_data.copy()
                batch_label_conf_data[index] = label_data.copy()
            
            valid_lsep_loss += sess.run(lsep_loss_op, feed_dict = {input_var : batch_image_data, label_conf_var : batch_label_conf_data, is_training : False})

        valid_lsep_loss /= valid_iteration
        if best_valid_lsep_loss > valid_lsep_loss:
            best_valid_lsep_loss = valid_lsep_loss
            saver.save(sess, './model/Inception_ResNet_v2_{}.ckpt'.format(iter))

        log_print('[i] valid lsep loss : {:.4f}, best valid lsep loss : {:.4f}'.format(valid_lsep_loss, best_valid_lsep_loss))

saver.save(sess, './model/Inception_ResNet_v2.ckpt')

for train_thread in train_threads:
    train_thread.end = False
    train_thread.join()
