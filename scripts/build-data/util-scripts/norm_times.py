import numpy as numpy
import pandas as pd 
import os
from os.path import join
from pdb import set_trace

wd = os.getcwd()
trial_dirs = []
target_file = 'game-info.csv'
hours = 4 # number hours off by

for i in range(1, 10):
     trial_dirs.append("trial-{}".format(i))
     
for d in os.listdir(wd):
    if d in trial_dirs:
        target_dir = join(wd, d)
    else:
        continue
    target_path = join(target_dir, target_file) 
    for f in os.listdir(target_dir):
        if f == target_file:
            target_df = pd.read_csv(target_path, header=None, index_col=None)
            target_df[1:] -=  hours*3600
            set_trace()
            target_df.to_csv(path_or_buf=target_path, header=None, 
                                index=False, float_format='%10f')