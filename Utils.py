import glob
import json
import numpy as np
import xml.etree.ElementTree as ET

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def read_txt(txt_path, replace_dir):
    json_paths = []
    for line in open(txt_path):
        json_paths.append(line.strip().replace('./json/', replace_dir))

    return json_paths

def get_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
    
def one_hot(label, classes):
    vector = np.zeros(classes, dtype = np.float32)
    vector[label] = 1.
    return vector

def one_hot_multi(labels, classes):
    vector = np.zeros(classes, dtype = np.float32)
    vector[labels] = 1.
    return vector

def Precision_Recall_Tags(pred_tags, gt_tags):
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0
    
    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]
    
    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100
    mAP = (precision + recall) / 2

    return precision, recall, mAP

if __name__ == '__main__':
    precision, recall, mAP = Precision_Recall_Tags(['Sky', 'Ocean', 'Person'],
                                                   ['Rain', 'Person', 'Car', 'Sky'])
    print(precision, recall, mAP)
