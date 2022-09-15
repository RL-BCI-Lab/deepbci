import os
import random
from pathlib import Path
from os.path import join 
from pdb import set_trace

import pandas as pd
import numpy as np
import tensorflow as tf

import directory_structure as ds

import deepbci.utils.logger as logger

def set_all_seeds(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def generate_experiments_summary(exp_paths):
    """ Collect all experiment metric and summary data to be compressed into a single files. 
    
        Assumes that the directory structure follows the directory structure generated
        by the dir_struct.py file.
    """
    trn_dfs = []
    tst_dfs = []
    tst_summary_dfs = []
    
    # Post experiments result processing of metrics
    for exp_path in exp_paths:
        exp_path = Path(exp_path)
        summarize = []
        trn_tst_dirs = get_train_and_test_dir(exp_path)
        if trn_tst_dirs is not None:
            summarize.append((exp_path, trn_tst_dirs))
        else:
            for path in Path(exp_path).iterdir():
                trn_tst_dirs = get_train_and_test_dir(path)
                if trn_tst_dirs is not None:
                    summarize.append((path, trn_tst_dirs))

        for path, metrics_paths in summarize:
            trn_metrics_df = pd.read_csv(metrics_paths['trn'])
            tst_metrics_df = pd.read_csv(metrics_paths['tst'])
            tst_metrics_summary_df = pd.read_csv(metrics_paths['tst_summary'])

            # Add experiment name to each row
            exp_name = os.path.basename(path)
            trn_metrics_df.insert(0, 'exp', exp_name)
            tst_metrics_df.insert(0, 'exp', exp_name)
            tst_metrics_summary_df.insert(0, 'exp', exp_name)
            
            trn_metrics_df.set_index(['exp', 'group', 'dataset', 'subject', 'trial'], inplace=True)
            tst_metrics_df.set_index(['exp', 'dataset', 'subject', 'trial'], inplace=True)
            
            trn_dfs.append(trn_metrics_df)
            tst_dfs.append(tst_metrics_df)
            tst_summary_dfs.append(tst_metrics_summary_df)
            
    parent_path = exp_path.parent
    trn_metrics_path = join(parent_path, 'train-metrics.csv')
    tst_metrics_path = join(parent_path, 'test-metrics.csv')
    tst_summary_path = join(parent_path, 'test-metrics-summary.csv')
    
    trn_dfs = pd.concat(trn_dfs)
    trn_dfs.sort_values(['exp', 'group', 'dataset', 'subject', 'trial'], inplace=True)
    trn_dfs.to_csv(trn_metrics_path, mode='w', header=True, index=True)
    
    tst_dfs = pd.concat(tst_dfs)
    tst_dfs.sort_values(['exp', 'dataset', 'subject', 'trial'], inplace=True)
    tst_dfs.to_csv(tst_metrics_path, mode='w', header=True, index=True)

    tst_summary_dfs = pd.concat(tst_summary_dfs)
    tst_summary_dfs.sort_values(['sub-exp', 'exp'], inplace=True)
    tst_summary_dfs.to_csv(tst_summary_path, mode='w', header=True, index=False)
    
def get_train_and_test_dir(exp_path):
    metric_paths = {}
    if exp_path.joinpath(ds.TRN_DIR).exists() and exp_path.joinpath(ds.TST_DIR).exists():
        metric_paths['trn'] = exp_path.joinpath(ds.TRN_DIR, 
                                                ds.RESULTS_DIR, 
                                                ds.METRICS_FILE)
        metric_paths['tst'] = exp_path.joinpath(ds.TST_DIR, 
                                                ds.RESULTS_DIR, 
                                                ds.METRICS_FILE)
        metric_paths['tst_summary'] = exp_path.joinpath(ds.TST_DIR, 
                                                        ds.RESULTS_DIR, 
                                                        ds.METRICS_SUMMARY_FILE)
        
        return metric_paths
