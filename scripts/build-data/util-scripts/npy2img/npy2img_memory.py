import os
import sys
import subprocess
from os.path import join 
from pdb import set_trace

import cv2
import numpy as np

from deepbci.utils.compress import zstd_decompress
from deepbci.utils.loading import get_trial_paths, path_to
from deepbci.utils.utils import parse_trials, timeme, rgb2gray

def save_image(path, image):
    file_name = "npy2img_memory_test.jpg"
    save_loc = join(path, file_name)
    cv2.imwrite(save_loc, np.squeeze(image))

def decompress_zstd(file_path):
    # Attempt to decompress file
    try:
        zstd_decompress(zstd_path=file_path)
    except Exception:
        print("Decompression failed!")

# VERY SLOW 
def load_npy(file_path):
    tmp_imgs = []

    print("Loading all images into memory...")
    for i in np.load(file_path):
        img = rgb2gray(i)
        img = cv2.resize(img, (84, 84))
        # save_image(os.path.dirname(f), img)
        tmp_imgs.append(i)

    return tmp_imgs
    
@timeme
def npy2img_memory(file_paths, shape=None, gray_scale=False, clean=False):
    imgs = []
    for f in file_paths:
        zstd_path = "{}.zst".format(f)
        print("Checking if .npy file exists...")
        if os.path.exists(f):
            print("Found .npy file!")
            tmp_imgs = np.load(f)
            print(len(tmp_imgs), f)
            imgs.append(np.stack(tmp_imgs, axis=0))
                
        elif os.path.exists(zstd_path):
            print("Attemtping to decompress zstd file...")
            decompress_zstd(file_path=zstd_path)
            if os.path.exists(f):
                print("Found .npy file!")
                tmp_imgs = load_npy(file_path=f)
                print(len(tmp_imgs), f)
                imgs.append(np.stack(tmp_imgs, axis=0))
        else:
            raise FileNotFoundError("No .npy or .npy.zst file detected!") 
        
        if clean: 
            print("Removing {}".format(f))
            os.remove(f)

    return imgs

if __name__ == "__main__":
    task = 'oa'
    subject = 'S02'
    trials = [[1, 20]]
    state_folder = 'states'
    npy_file = 'state_imgs.npy'
    clean = True

    trials = parse_trials(trials)
    subject_path = path_to( 
            root_dir=os.getcwd(), target=join(task, subject))
    trial_paths = get_trial_paths(
            dir_path=subject_path, trial_numbers=trials)

    for i, t in enumerate(trial_paths):
        trial_paths[i] = join(t, state_folder, npy_file)

    imgs = npy2img_memory(file_paths=trial_paths, clean=True)
    print(sys.getsizeof(imgs))
    set_trace()