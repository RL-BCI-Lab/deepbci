# Running BCI Experiments
These are instructions for how to conduct BCI experiments for all games contained with the deebci/games/ directory. This section assumes that you have the built or pulled the Docker image mentioned above.

1. Start Docker container 
`docker start dbci`

2. Attach to Docker container
`docker attach dbci`

3. Make sure the working directory is mnt/ (this should be the default directory when attaching to the container). For the remaining steps before running the experiments record the impedance levels given in the OpenBCI GUI. After running the experiments record the and task related data that is asked for in the experiment spread sheets. Additionally, check the command-line output to make sure no vast amounts of samples were dropped (1-10 samples is acceptable over 50-100 dropped samples in a row means the trial needs to be rerecorded). 

4. Run 5 trials of BGSObs and be sure to use correct subject and trial number.

`python deepbci/games/binary_goal_search/observation/ -s <subject number> -t <trial number>`

5. Run 5 trials of BGSInt and be sure to use correct subject and trial number.

`python deepbci/games/binary_goal_search/interaction/ -s <subject number> -t <trial number>`

6. Check if subject needs a break. If so provide a 10-15 minute break and prepare to setup EEG equipment.

7. Run 5 trials of OAObs and be sure to use correct subject and trial number.

8. Run 5 trials of OAOut and be sure to use correct subject and trial number. Additionally, be sure to set seed based on trial number inside config.

```
Trial:  1   2  3  4 5  6  7   8   9   10  11  12  13  14  15  16
Seeds:  0, 1, 2, 4, 6, 7, 11, 12, 17, 30, 31, 32, 33, 35, 37, 100
```

9. Check if subject needs a break. If so provide a 10-15 minute break and prepare to setup EEG equipment.

10. Repeat steps 4-9 once

11. Clean up