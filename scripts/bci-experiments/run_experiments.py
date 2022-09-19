import os
from pdb import set_trace

import numpy as np

from deepbci.utils import utils

# Frequently change variables
subject_number = 0 # Change per session
trial_range = (1, 5) # Change per session

# Order in which tasks will be ran
task_order = [
    ('binary_goal_search', 'observation'),
    ('binary_goal_search', 'interaction'),
    ('obstacle_avoidance', 'observation'),
    ('obstacle_avoidance', 'outcome'),
]

# Trial numbers
trials = np.arange(trial_range[0], trial_range[1]+1)

dbci_path = utils.get_module_root()

# Commands and flag templates 
cmd_template = "python {}/games/{}/{}/ {}"
default_flags = "-s {} -t {}"
unique_flags = {
    ('obstacle_avoidance', 'observation'): " --seed {}"
}
unique_flag_values = {
    ('obstacle_avoidance', 'observation'): [
         0, 1, 2, 4, 6, 7, 11, 12, 17, 30, 31, 32, 33, 35, 37, 100
    ]
}

def run():
    for task, sub_task in task_order:
        for trial_number in trials:
            
            # Get flags
            if (task, sub_task) in unique_flags:
                flags = default_flags + unique_flags[(task, sub_task)]
                unique_values = unique_flag_values[(task, sub_task)][trial_number-1]
                flags = flags.format(subject_number, trial_number, unique_values)
            else:
                flags = default_flags.format(subject_number, trial_number)
            
            # Build cmd to run
            cmd = cmd_template.format(dbci_path, task, sub_task, flags)
            print('-'*50)
            print("Running cmd: {}".format(cmd))
            print('-'*50)
            
            # Enter 'c' to move run trial
            set_trace() 
            
            # Run cmd
            os.system(cmd)
            
            # Log information again
            print('-'*50)
            print("Finished running cmd: {}".format(cmd))
            print('-'*50)
            


if __name__ == "__main__":
    run()