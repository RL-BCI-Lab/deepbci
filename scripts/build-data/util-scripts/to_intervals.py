import numpy as np
import pandas as pd
import os
from os.path import join
from pdb import set_trace

wd = os.getcwd()
targets = []
rec_rate = .1
crn_file = 'time-crn-obs-state-{}.csv'
ern_file = 'time-ern-obs-state-{}.csv'

for i in range(1, 21):
 targets.append("trial-{}".format(i))

for d in os.listdir(wd):
    if any(d ==  t for t in targets):
        target_dir = join(wd, d)
        if len(d) == 7:
            trial = d[-1]
        elif len(d) == 8:
            trial = d[-2:]
        print("FOUND: {} {}".format(target_dir, len(d)))
        crn_target = join(target_dir, crn_file.format(trial))
        ern_target = join(target_dir, ern_file.format(trial)) 

        crn_df = pd.read_csv(crn_target, header=None, index_col=None)
        ern_df = pd.read_csv(ern_target, header=None, index_col=None)
        interval_df = pd.concat([crn_df, ern_df], ignore_index=True)
        intervals = np.sort(np.array(interval_df), axis=0)
        save_file = os.path.join(target_dir, "time-interval.csv")
        print(len(crn_df) + len(ern_df), intervals.shape)
        print(d, target_dir)
        set_trace()
        np.savetxt(save_file, intervals, fmt='%1f', delimiter=',')