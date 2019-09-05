
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

from loss.LSEP_Loss import *
from loss.TopK_Loss import *
from loss.Focal_Loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
helper = NUS_WIDE_Helper(LABEL_TXT_PATH)

test_json_paths = read_txt('./dataset/test.txt', REPLACE_JSON_DIR)

# open('./log.txt', 'w').close()
print('[i] Test : {}'.format(len(test_json_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
predict_dic = Image_Tagging(input_var, False, CLASSES, decision_model = True)

# 3. test
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/Inception_ResNet_v2_Decision_110000.ckpt')

test_threshold = 0.40
test_iteration = len(test_json_paths) // BATCH_SIZE

test_mAP_list = []
test_precision_list = []
test_recall_list = []

batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

for i in range(test_iteration):
    batch_gt_tags = []
    batch_json_paths = test_json_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

    for index, json_path in enumerate(batch_json_paths):
        image_data, gt_tags = helper.get_test_data(json_path, get_tags = True)
        batch_image_data[index] = image_data.copy()
        batch_gt_tags.append(gt_tags)
    
    batch_pred_confs, batch_pred_counts = sess.run([predict_dic['Confidence'], predict_dic['TopK']], feed_dict = {input_var : batch_image_data})

    # for pred_conf, pred_count, gt_tags in zip(batch_pred_confs, batch_pred_counts, batch_gt_tags):
    #     topk = np.argmax(pred_count) + 1
    #     pred_indexs = np.argsort(pred_conf)[::-1]
    #     pred_indexs = pred_indexs[:topk]
    #     pred_indexs = np.asarray([pred_index for pred_index in pred_indexs if pred_conf[pred_index] >= test_threshold], dtype = np.int32)

    #     pred_tags = helper.class_names[pred_indexs]

    #     test_precision, test_recall, test_mAP = Precision_Recall_Tags(pred_tags, gt_tags)

    #     test_mAP_list.append(test_mAP)
    #     test_precision_list.append(test_precision)
    #     test_recall_list.append(test_recall)

    # sys.stdout.write('\r[{}/{}] precision = {:.2f}%, recall = {:.2f}%'.format(i + 1, test_iteration, np.mean(test_precision_list), np.mean(test_recall_list)))
    # sys.stdout.flush()
    
    for image, pred_conf, pred_count, gt_tags in zip(batch_image_data, batch_pred_confs, batch_pred_counts, batch_gt_tags):
        topk = np.argmax(pred_count) + 1
        pred_indexs = np.argsort(pred_conf)[::-1]
        pred_indexs = pred_indexs[:topk]
        pred_indexs = np.asarray([pred_index for pred_index in pred_indexs if pred_conf[pred_index] >= test_threshold], dtype = np.int32)

        pred_tags = helper.class_names[pred_indexs]
        
        print('## Prediction')
        for name, prob in zip(pred_tags, pred_conf[pred_indexs]):
            print('# {:30s} : {:.2f}%'.format(name, prob * 100))
        print()
        
        print('## Ground Truth')
        for name in gt_tags:
            print('# {:20s}'.format(name))
        print()

        cv2.imshow('show', image.astype(np.uint8))
        cv2.waitKey(0)

print()

test_mAP = np.mean(test_mAP_list)
log_print('[i] test_mAP : {:.2f}'.format(test_mAP))

# NUS-WIDE 1K (Test set)
# Precision = 30.74%
# Recall = 21.52%
# mAP = 26.13%
