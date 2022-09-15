import os
import fnmatch
from os.path import join
from pdb import set_trace

import numpy as np
import pandas as pd
# Run script from within scripts directory to work!
wd = os.path.dirname(os.getcwd())
state_file = 'state-info.csv'
oa = ['timestamps', 'labels', 'actions']
bgs = ['timestamps', 'labels', 'actions', 'terminal']

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def group_consecutive_labels(labels):
    label_loc = np.where(labels == 1)[0]
    group_split_condi = np.where(np.diff(label_loc) != 1)[0]+1
    groups = np.split(label_loc, group_split_condi)
    first_locs = [g[0] for g in groups]
    other_locs = np.hstack([g[1:] for g in groups])
    new_labels = np.zeros(labels.shape, dtype=int)
    new_labels[first_locs] = 1
    
    mask = labels == new_labels
    mask_false_locs = np.where(mask == False)[0]
    assert (mask_false_locs == other_locs).all()
    
    return new_labels 

for state_path in find_files(wd, state_file):
    state_df = pd.read_csv(state_path, header=0, index_col=None)
    
    if set(oa) == set(state_df.columns):
        rewards = group_consecutive_labels(state_df.labels.values)
        reward_df = pd.DataFrame(rewards, columns=['rewards'])
    else:
        continue
    
    print("Updated {}".format(state_path))
    to_concat = [state_df.timestamps, state_df.labels, reward_df.rewards, state_df.actions]
    new_state_df = pd.concat(to_concat, axis=1)
    # one_locs = new_state_df.index[new_state_df['labels'] == 1].tolist()
    new_state_df.to_csv(state_path, index=False, float_format='%6f')
