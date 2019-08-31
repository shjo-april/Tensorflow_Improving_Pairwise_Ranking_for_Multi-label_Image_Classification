
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
helper = None # NUS_WIDE_Helper(LABEL_TXT_PATH)

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
label_topk_var = tf.placeholder(tf.float32, [None, MAX_TOP_K])
is_training = tf.placeholder(tf.bool)

print('[i] generate graph !')
predict_dic = Image_Tagging(input_var, is_training, CLASSES, decision_model = True)

vars = tf.trainable_variables()

lsep_vars = []
decision_vars = []

for var in vars:
    if 'Label_Decision' in var.name:
        decision_vars.append(var)
    else:
        lsep_vars.append(var)

lsep_loss_op = Log_Sum_Exp_Pairwise_Loss(predict_dic['LSEP'], label_conf_var, BATCH_SIZE)
topk_loss_op = TopK_Loss(predict_dic['TopK'], label_topk_var, BATCH_SIZE)
conf_loss_op = Focal_Loss(predict_dic['Confidence'], label_conf_var)
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in decision_vars]) * WEIGHT_DECAY

loss_op = lsep_loss_op + topk_loss_op + conf_loss_op + l2_reg_loss_op

tf.summary.scalar('Loss', loss_op)
tf.summary.scalar('LSEP_Loss', lsep_loss_op)
tf.summary.scalar('TopK_Loss', topk_loss_op)
tf.summary.scalar('Focal_Loss', conf_loss_op)
tf.summary.scalar('L2_Regularization_Loss', l2_reg_loss_op)
summary_op = tf.summary.merge_all()
print('[i] set LSEP/TopK/Confidense loss op !')

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    print('[i] ready Optimizer', end = '')
    train_op = tf.train.MomentumOptimizer(learning_rate_var, 0.9).minimize(loss_op, var_list = decision_vars)
    print('\r[i] set Optimizer')

# 3. train
print('[i] ready tf.Session()', end = '')
sess = tf.Session() 
print('\r[i] set tf.Session()       ')

print('[i] ready Session !', end = '')
sess.run(tf.global_variables_initializer())
print('\r[i] set global_variables_initializer !')

# '''
pretrained_saver = tf.train.Saver(var_list = lsep_vars)
pretrained_saver.restore(sess, './model/VGG16_50000.ckpt')
log_print('[i] restored LSEP Loss (VGG16)')
# '''

saver = tf.train.Saver()
# saver.restore(sess, './model/VGG16_Decision_50000.ckpt')

print('[i] ready thread')
for index, train_thread in enumerate(train_threads):
    print('[i] thread.start() {}'.format(index))
    train_thread.start()
print('[i] start all thread')

max_epoch = 50
learning_rate = INIT_LEARNING_RATE
max_iteration = len(train_threads[0].json_paths) // BATCH_SIZE * max_epoch
decay_iteration = np.asarray([0.5 * max_iteration, 0.75 * max_iteration], dtype = np.int32)

valid_threshold = 0.1
best_valid_mAP = 0.0
valid_iteration = len(valid_json_paths) // BATCH_SIZE

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
lsep_loss_list = []
topk_loss_list = []
conf_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train_Label_Decision')

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
    
    _feed_dict = {input_var : batch_image_data, label_topk_var : batch_label_topk_data, label_conf_var : batch_label_conf_data, is_training : True, learning_rate_var : learning_rate}
    _, loss, lsep_loss, topk_loss, conf_loss, l2_reg_loss, summary = sess.run([train_op, loss_op, lsep_loss_op, topk_loss_op, conf_loss_op, l2_reg_loss_op, summary_op], feed_dict = _feed_dict)
    
    if np.isnan(loss):
        print('[!]', loss, lsep_loss, topk_loss, conf_loss, l2_reg_loss)
        input()
    
    loss_list.append(loss)
    lsep_loss_list.append(lsep_loss)
    topk_loss_list.append(topk_loss)
    conf_loss_list.append(conf_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    train_writer.add_summary(summary, iter)
    
    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        lsep_loss = np.mean(lsep_loss_list)
        topk_loss = np.mean(topk_loss_list)
        conf_loss = np.mean(conf_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)

        log_print('[i] iter : {}, loss : {:.4f}, lsep_loss : {:.4f}, topk_loss : {:.4f}, conf_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, lsep_loss, topk_loss, conf_loss, l2_reg_loss, train_time))

        loss_list = []
        lsep_loss_list = []
        topk_loss_list = []
        conf_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        valid_precision_list = []
        valid_recall_list = []

        batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
        
        for i in range(valid_iteration):
            batch_gt_tags = []
            batch_json_paths = valid_json_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            for index, json_path in enumerate(batch_json_paths):
                image_data, gt_tags = helper.get_test_data(json_path, get_tags = True)

                batch_image_data[index] = image_data.copy()
                batch_gt_tags.append(gt_tags)
            
            batch_pred_confs, batch_pred_counts = sess.run([predict_dic['Confidence'], predict_dic['TopK']], feed_dict = {input_var : batch_image_data, is_training : False})

            for pred_conf, pred_count, gt_tags in zip(batch_pred_confs, batch_pred_counts, batch_gt_tags):
                topk = np.argmax(pred_count) + 1
                pred_indexs = np.argsort(pred_conf)[::-1]
                pred_indexs = pred_indexs[:topk]
                pred_indexs = np.asarray([pred_index for pred_index in pred_indexs if pred_conf[pred_index] >= valid_threshold], dtype = np.int32)

                pred_tags = helper.class_names[pred_indexs]

                valid_precision, valid_recall, valid_mAP = Precision_Recall_Tags(pred_tags, gt_tags)

                valid_precision_list.append(valid_precision)
                valid_recall_list.append(valid_recall)

            sys.stdout.write('\r[{}/{}] precision = {:.2f}%, recall = {:.2f}%'.format(i + 1, valid_iteration, np.mean(valid_precision_list), np.mean(valid_recall_list)))
            sys.stdout.flush()
        print()

        valid_precision, valid_recall = np.mean(valid_precision_list), np.mean(valid_recall_list)
        valid_mAP = (valid_precision + valid_recall) / 2

        if best_valid_mAP < valid_mAP:
            best_valid_mAP = valid_mAP

            log_print('[i] valid precision : {:.2f}'.format(valid_precision))
            log_print('[i] valid recall : {:.2f}'.format(valid_recall))
            saver.save(sess, './model/VGG16_Decision_{}.ckpt'.format(iter))

        log_print('[i] valid mAP : {:.2f}, best valid mAP : {:.2f}%'.format(valid_mAP, best_valid_mAP))

saver.save(sess, './model/VGG16_Decision.ckpt')

for train_thread in train_threads:
    train_thread.end = False
    train_thread.join()
