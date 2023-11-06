import cv2
import numpy as np

import hyperparameters


def read_im(path, bit_depth=16):
    scale = (2**bit_depth) - 1  # 255 for 8-bit
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if im == None:
        raise FileNotFoundError
    return im / scale


def random_sample_patch(low_im: np.array, high_im: np.array, crop_size: int, threshold: float):
    assert low_im.shape == high_im.shape
    h, w = low_im.shape
    if h <= crop_size:
        h = crop_size + 1
        low_im = cv2.resize(low_im, (w, h)) # why does this work like this ???
        high_im = cv2.resize(high_im, (w, h))
    if w <= crop_size:
        w = crop_size + 1
        low_im = cv2.resize(low_im, (w, h))
        high_im = cv2.resize(high_im, (w, h))
    
    # find a random crop with some details and shapes
    std_dev = 0
    trial = hyperparameters.sample_trial
    while std_dev < threshold and trial > 0:
        trial -= 1  # to avoid inf loop
        x = np.random.randint(0, w - crop_size)
        y = np.random.randint(0, h - crop_size)
    
        low_crop = low_im[y : y + crop_size, x : x + crop_size]
        high_crop = high_im[y : y + crop_size, x : x + crop_size]
        std_dev = max(low_crop.std(), high_crop.std())
    
    
    low_crop = np.expand_dims(low_crop, 0)
    high_crop = np.expand_dims(high_crop, 0)
    return low_crop.astype(np.float32), high_crop.astype(np.float32)