"""
    Script postprocessing experiment related information. Useful when generating
    additional files after experiment recording sessions, updating names, or fixing
    trivial errors.
"""
import argparse
import os
import sys
import fnmatch
import json
from pathlib import Path
from os.path import join
from pdb import set_trace

import numpy as np
import pandas as pd
import cv2

from deepbci.utils.compress import zstd_decompress

STATE_INFO_FILE = 'state-info.csv'
RESTING_FILE = "time-rest.csv"
STATE_IMAGES_FILE = "state-images.npy.zst"
OLD_STATE_IMAGES_FILE = "human-state-images.npy"

def search(directory, target):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, target):
                filename = os.path.join(root, basename)
                yield filename

def _find_files(dir_path, target):
    paths = []
    for state_path in search(dir_path, target):
        paths.append(Path(state_path))
    if len(paths) == 0:
        print(f"{target} not found in {dir_path}")
    return paths

def build_resting_events(dir_path, dry_run=False):
    state_info_paths = _find_files(dir_path, target=STATE_INFO_FILE)
    
    for state_path in state_info_paths:
        parent = state_path.parent
        resting_file_path = join(parent, RESTING_FILE)
        
        state_df = pd.read_csv(state_path, header=0, index_col=None)
        resting_ts = state_df['timestamps'][state_df['labels'] == 0]
        
        print(f"Total resting states: {len(resting_ts)}")
        print(f"Saving resting state to: {resting_file_path}")
        if not dry_run:
            resting_ts.to_csv(resting_file_path, index=False, header=False)
        else:
            print(f"Timestamps found:\n{resting_ts}")

def rename_state_images(dir_path, new_file_name, target=STATE_IMAGES_FILE, dry_run=False):
    state_image_file_paths = _find_files(dir_path, target)
    print(f"Renaming found files to {target}...")
    for image_path in state_image_file_paths:
        parent = image_path.parent
        new_file_path = image_path.parent / new_file_name
        print(f"Renaming file at: {parent}")
        print(f"Renaming {image_path.name} -> {new_file_name}")
        if not dry_run:
            image_path.rename(new_file_path)

def npy2image(dir_path, target=STATE_IMAGES_FILE, ext='png', dry_run=False):
    
    state_image_file_paths = _find_files(dir_path, target)

    for image_path in state_image_file_paths:
        parent = image_path.parent
        try:
            zstd_decompress(zstd_path=image_path)
        except Exception:
            print("Decompression failed!")
        npy_path = parent / image_path.stem
        images = np.load(npy_path)

        print(f"File contains {len(images)} images")
        try:
            if not dry_run:
                for f, i in enumerate(images):
                    if i.shape[-1] == 3:
                        image = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                    else:
                        image = i
                        
                    image_path = parent / '{}{}.{}'.format('state_', f, ext)
                    cv2.imwrite(str(image_path), image)
        finally:
            os.remove(npy_path)
            
def kwarg_parser(kv):
    key, value = kv.split("=", 1)
    return key, value

if __name__ == '__main__':
    module = sys.modules[__name__]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True,
                        help="directory path that will be searched to find all state-info.csv files")
    parser.add_argument('--func', type=str, required=True,
                         help="name of function to run within state_info.py script")
    parser.add_argument("--kwargs", action='append', type=kwarg_parser, default={})
    parser.add_argument('--dry-run', action='store_true', 
                    help="dry run function without making real modifications")
    args = parser.parse_args()

    if hasattr(module, args.func):
        func = getattr(module, args.func)
    else:
        raise AttributeError(f"No function {args.func} found.")

    func(dir_path=args.dir, dry_run=args.dry_run, **dict(args.kwargs))
