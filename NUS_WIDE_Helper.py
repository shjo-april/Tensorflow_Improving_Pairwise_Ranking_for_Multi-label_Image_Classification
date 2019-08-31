
import cv2
import numpy as np

from Utils import *
from Define import *
from DataAugmentation import *

class NUS_WIDE_Helper:

    class_dic = {}
    class_names = []
    
    def __init__(self, label_txt_path):
        self.class_names = [line.strip() for line in open(label_txt_path).readlines()]
        self.class_names = np.asarray(self.class_names)
        
        self.class_dic = {class_name : i for i, class_name in enumerate(self.class_names)}
    
    def get_data(self, json_path):
        image_path = json_path.replace('/json/', '/image/')[:-4] + 'jpg'
        
        image = cv2.imread(image_path)
        image = DataAugmentation(image)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image_data = image.astype(np.float32)

        labels = [self.class_dic[tag_name] for tag_name in get_json_data(json_path)['Tags']]

        label_topk_data = one_hot(min(len(labels) - 1, MAX_TOP_K - 1), MAX_TOP_K)
        label_conf_data = one_hot_multi(labels, CLASSES)
        
        return image_data, label_topk_data, label_conf_data
        
    def get_test_data(self, json_path, get_tags):
        image_path = json_path.replace('/json/', '/image/')[:-4] + 'jpg'
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image_data = image.astype(np.float32)
        
        if get_tags:
            tags = get_json_data(json_path)['Tags']
            return image_data, tags

        else:
            labels = [self.class_dic[tag_name] for tag_name in get_json_data(json_path)['Tags']]
            # label_topk_data = one_hot(min(len(labels) - 1, MAX_TOP_K - 1), MAX_TOP_K)
            label_conf_data = one_hot_multi(labels, CLASSES)
            return image_data, label_conf_data