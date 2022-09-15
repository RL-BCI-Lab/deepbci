import numpy as np
import pandas as pd
import os
from pdb import set_trace

wd = os.getcwd()
targets = []
rec_rate = .1
game_info = "game-info"

for i in range(1, 21):
 targets.append("trial-{}".format(i))

for d in os.listdir(wd):
    if any(d ==  t for t in targets):
        target_dir = os.path.join(wd, d)
        print("FOUND: {}".format(target_dir))
        for f in os.listdir(target_dir):
            if f.startswith (game_info):
                target_game = os.path.join(target_dir, f)
                print("\tFOUND {}".format(target_game))
                game_dur = pd.read_csv(target_game, header=None, index_col=None).iloc[0][0]

        # generate intervals
        state_intervals =  np.arange(0 , game_dur, rec_rate)
        print(game_dur, len(state_intervals), state_intervals[-1])

        save_file = os.path.join(target_dir, "time-interval.csv")
        set_trace()
        np.savetxt(save_file, np.vstack(state_intervals), fmt='%1f', delimiter=',') 
