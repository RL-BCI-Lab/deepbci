"""
Static variables for building exp/ directory structure (advised not to change).

All scripts within this directory will refer to this file for directory and file names.
"""

import os
from os.path import join
from pathlib import Path


# Working directory path (i.e. path path to this file)
# Note: When ran on HPC an error will be thrown if this is not cast as a string
ROOT_DIR = str(Path(__file__).parent.absolute())

# Root directory for experiment results to be saved to
EXPS_DIR = 'exps'

# Default config directory 
CFG_DIR = 'configs'

# Default cache directory and data pickle file name
CACHE_DIR = 'cache'
CACHE_FILE = 'data-cache'

# Default Sub-directories for storing all training and testing related files.
TRN_DIR = 'train'
TST_DIR = 'test'

# Default tensorboard sub-directory and sub-file names
TENSORBOARD_DIR = 'tensorboard'

# Default checkpoint sub-directory and sub-file names
CKPT_DIR = 'checkpoints'
CKPT_FILE = 'model.ckpt'

# Default scaler sub-file name
# SCALER_FILE = 'scaler' # dill/pickle file no extension

# Default metric sub-directory and sub-file names
RESULTS_DIR = 'results'
METRICS_FILE = 'metrics.csv'
METRICS_SUMMARY_FILE = 'metrics-summary.csv'

# Default prediction sub-directory and file names
PREDS_DIR = join(RESULTS_DIR, 'predictions')
TRN_PRED_FILE = 'train-preds.csv'
VLD_PRED_FILE = 'valid-preds.csv'
TST_PRED_FILE = '{}.csv'

# Default layer output sub-directory and file names
LAYER_OUTPUTS_DIR = join(RESULTS_DIR, 'layer-outputs')
LAYER_OUTPUTS_FILE = '{}'

# Default sub-file names for saving configs 
DATA_CFG = 'data.yaml'
MODEL_CFG = 'model.yaml'

# Default sub-directory for any experiment ran without a exp_dir variable
# TRAIN_EXP_DIR = 'train-exp'
LOGO_EXP_DIR = 'logocv'

# Default names for configs used in run_exp.py 
EXP_FILE = 'exp.yaml'
EXP_DEF_FILE = 'exp-def.yaml'


def generate_directory_structure():
    output_dir_path = join(ROOT_DIR, EXPS_DIR)
    cfg_dir_path = join(ROOT_DIR, CFG_DIR)
    cache_dir_path = join(ROOT_DIR, CACHE_DIR)
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        
    if not os.path.exists(cfg_dir_path):
        os.makedirs(cfg_dir_path)

    if not os.path.exists(cache_dir_path):
        os.makedirs(cache_dir_path)

if __name__ == '__main__':
    generate_dir_structure()