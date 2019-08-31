import cv2
import random

import numpy as np

# prob = 50%
def random_flip(image, condition = [False, True]):
    if random.choice(condition):
        h, w, c = image.shape
        image = np.fliplr(image).copy()

    return image

# prob = 50%
def random_scale(image, condition = [False, True]):
    if random.choice(condition):
        h, w, c = image.shape

        w_scale = random.uniform(0.8, 1.2)
        h_scale = random.uniform(0.8, 1.2)

        image = cv2.resize(image, (int(w * w_scale), int(h * h_scale)))

    return image

# prob = 20%
def random_blur(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        f = random.choice([3, 5])
        image = cv2.blur(image, (f, f))
    return image

# prob = 20%
def random_brightness(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        adjust = random.uniform(0.8, 1.2)
        image = np.clip(image.astype(np.float32) * adjust, 0, 255).astype(np.uint8)

    return image

# prob = 20%
def random_hue(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        h = h.astype(np.float32)
        adjust = random.uniform(0.8, 1.2)

        h = np.clip(h * adjust, 0, 255).astype(np.uint8)

        hsv_image = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

# prob = 20%
def random_saturation(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        s = s.astype(np.float32)
        adjust = random.uniform(0.8, 1.2)

        s = np.clip(s * adjust, 0, 255).astype(np.uint8)

        hsv_image = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

# prob = 5%
def random_gray(image, condition = [True] + [False] * 19):
    if random.choice(condition):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.merge([image, image, image])
    return image

# prob = 25%
def random_crop(image, condition = [False, False, False, True]):
    if random.choice(condition):
        h, w, _ = image.shape

        random_xmin = random.randint(0, int(w * 0.05))
        random_ymin = random.randint(0, int(h * 0.05))
        
        random_xmax = random_xmin + random.randint(int((w - random_xmin) * 0.8), w - random_xmin)
        random_ymax = random_ymin + random.randint(int((h - random_ymin) * 0.8), h - random_ymin)

        image = image[random_ymin : random_ymax, random_xmin : random_xmax]

    return image

def DataAugmentation(image):
    image = random_flip(image)
    image = random_scale(image)
    image = random_blur(image)
    image = random_brightness(image)
    image = random_saturation(image)
    # image = random_gray(image)
    image = random_crop(image)

    return image

if __name__ == '__main__':
    image_path = 'D:/DB/COCO/train2017/image/000000000009.jpg'
    
    image = cv2.imread(image_path)

    cv2.imshow('original', image)
    cv2.waitKey(1)
    
    while True:
        aug_image = DataAugmentation(image)
        cv2.imshow('DataAugmentation', aug_image)
        cv2.waitKey(0)
