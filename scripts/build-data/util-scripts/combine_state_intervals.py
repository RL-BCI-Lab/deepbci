import os
import fnmatch
from os.path import join
from pdb import set_trace

import numpy as np
import pandas as pd
# Run script from within scripts directory to work!
wd = os.path.dirname(os.getcwd())
int_file = 'time-interval.csv'
state_file = 'state-info.csv'
oa = ['targets', 'actions']
bgs = ['targets', 'actions', 'terminal']
new_oa = ['labels', 'actions']
new_bgs = ['labels', 'actions', 'terminal']

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

for f in find_files(wd, int_file):
    subj_dir = os.path.dirname(f)
    int_path = f
    state_path = join(subj_dir, state_file) 

    int_df = pd.read_csv(int_path, names=['timestamps'], index_col=None)
    state_df = pd.read_csv(state_path, header=0, index_col=None)
    
    if set(bgs).issubset(set(state_df.columns)):
        state_df = state_df[bgs]
        new_labels = new_bgs 
    elif set(oa).issubset(set(state_df.columns)):
        state_df = state_df[oa]
        new_labels = new_oa
        
    if len(int_df) != len(state_df):
        print( len(int_df), len(state_df))
        print("SKIPPED: {}".format(subj_dir))
        continue
    
    print("Updated {}".format(state_path))
    new_state_df = pd.concat([int_df, state_df], axis=1)
    new_state_df.columns = list(int_df.columns) + new_labels
    print(new_state_df.columns)
    new_state_df.to_csv(state_path, index=False, float_format='%6f')