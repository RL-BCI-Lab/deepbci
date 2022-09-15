import os
import subprocess

import cv2
import numpy as np

from deepbci.utils.compress import zstd_decompress

def npy2img(folder, save_path, rate=1, ext='png', dec=1, delete=False):
    zstd_path = "{}.zst".format(save_path)

    # Attempt to decompress file
    try:
        zstd_decompress(zstd_path=zstd_path)
    except Exception:
        print("Decompression failed!")

    # Attempt to extract images from .npy
    print("Checking if .npy file exists...")
    if os.path.exists(save_path):
        images = np.load(save_path)
        print("Found .npy file!")

        # Type and value checking for passed rate
        if isinstance(rate, int) or isinstance(rate, float):
            frames = np.round(np.arange(0, (len(images))/rate, 1/rate), dec)
        elif isinstance(rate, np.ndarray): 
            if len(rate) == len(images):
                raise ValueError("The number of rates and images do not match!")
            frames = rate
        else:
            raise TypeError("Invalid rate type passed!")

        # Create images loaded from .npy
        print("Attemtping to convert .npy to {} format...".format(ext))
        print("File containes {} images".format(len(images)))
        for i, f in zip(images, frames):
            if i.shape[-1] == 3:
                image = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            else:
                image = i
                
            image_path = os.path.join(folder, '{}{}.{}'.format('state_', f, ext))
            cv2.imwrite(image_path, image)
    else:
        raise Exception("No .npy file detected!")

    # Attempt to delete .npy file
    if delete: 
        print("Removing {}".format(save_path))
        os.remove(save_path)

# python -m bci.preprocessing.data.scripts.npy2img -t flappy -s 00 -tl 0 -e png -r 10 -c True
