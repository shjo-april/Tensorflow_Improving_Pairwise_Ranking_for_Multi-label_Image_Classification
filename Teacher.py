
import cv2
import random
import threading

import numpy as np

from Define import *
from Utils import *
from NUS_WIDE_Helper import *
from DataAugmentation import *

class Teacher(threading.Thread):
    end = False
    ready = False
    min_data_size = 1
    max_data_size = 50

    batch_size = 0

    batch_data_length = 0
    batch_data_list = []
    
    helper = None
    json_paths = []
    
    def __init__(self, txt_path, helper, min_data_size = 1, max_data_size = 50, batch_size = BATCH_SIZE):
        self.batch_size = batch_size
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.helper = helper
        self.json_paths = read_txt(txt_path, REPLACE_JSON_DIR)

        threading.Thread.__init__(self)

    def get_batch_data(self):
        batch_image_data, batch_label_topk_data, batch_label_conf_data = self.batch_data_list[0]

        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False

        return batch_image_data, batch_label_topk_data, batch_label_conf_data

    def run(self):
        while not self.end:
            while self.batch_data_length >= self.max_data_size:
                continue
            
            batch_image_data = []
            batch_label_topk_data = []
            batch_label_conf_data = []
            batch_json_paths = random.sample(self.json_paths, BATCH_SIZE * 2)
            
            for json_path in batch_json_paths:
                image_data, label_topk_data, label_conf_data = self.helper.get_data(json_path)

                batch_image_data.append(image_data)
                batch_label_topk_data.append(label_topk_data)
                batch_label_conf_data.append(label_conf_data)

                if len(batch_image_data) == BATCH_SIZE:
                    break
            
            batch_image_data = np.asarray(batch_image_data, dtype = np.float32)
            batch_label_topk_data = np.asarray(batch_label_topk_data, dtype = np.float32)
            batch_label_conf_data = np.asarray(batch_label_conf_data, dtype = np.float32)

            self.batch_data_list.append([batch_image_data, batch_label_topk_data, batch_label_conf_data])
            self.batch_data_length += 1
            
            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False
