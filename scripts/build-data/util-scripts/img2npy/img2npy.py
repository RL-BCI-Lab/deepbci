import sys
import os
import re
import traceback
import subprocess
from pdb import set_trace

import cv2
import numpy as np 

from utils.utils import rgb2gray
from utils.compress import zstd_compress

def tryfloat(s):
    try:
        return float(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of strings and number chunks.
        "state_0.1.BMP" -> ["state_", 0.1, ".BMP"]
    """

    for c in re.split("([0-9]+.[0-9]+)|([0])", s):
        if isinstance(tryfloat(c), float):
            return tryfloat(c) 
    return None

def img2npy(
    folder, save_path, ext, grayscale=False, resize=None, delete_imgs=False):
    rgb_imgs = []
    found = os.listdir(folder)
    print("Files/Folders Found:", len(found))
    for name in found:
        if alphanum_key(name) is None or not name.endswith('.'+ext):
            found.remove(name)
            print(name)
    print("Images found: {}".format(len(found)))
    sorted_imgs = sorted(found, key=alphanum_key)

    for img in sorted_imgs:
        img_path = os.path.join(folder, img)
        bgr_img = cv2.imread(img_path)
        b,g,r = cv2.split(bgr_img)
        img = cv2.merge([r,g,b]) # Translate from bgr to rgb
        
        if grayscale: 
            img = rgb2gray(img)
            
        if resize is not None:
            img = cv2.resize(img, tuple(resize))
            
        # Quick check by saving an image
        #cv2.imwrite(save_loc+'/test.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        rgb_imgs.append(img)
        
        if delete_imgs: 
            os.remove(img_path)

    print("Image stack size: {}".format(len(rgb_imgs))) 
    np.save(save_path, np.stack(rgb_imgs, axis=0))
    zstd_compress(file_path=save_path, clean=True)