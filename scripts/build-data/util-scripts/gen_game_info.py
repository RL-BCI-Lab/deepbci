import os 
import numpy as np 
import pandas as pd 
from pdb import set_trace

wd = os.getcwd()
targets = []
game_info = "game-info"
count_down = 30

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
                info = pd.read_csv(target_game, header=None, index_col=None).as_matrix()

        print("Game Duration: ", info[0])
        if len(info) == 4:
            print(info)
            continue
        start = info[1] + count_down
        end = info[0] + start
        info = np.append(info, [start, end]).reshape(-1, 1)

        print(target_game)
        print(info)
        set_trace()
        np.savetxt(target_game, info, fmt='%.6f', delimiter=',') 