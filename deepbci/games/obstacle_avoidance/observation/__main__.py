import argparse
import os
from os.path import join, dirname
from pdb import set_trace

import deepbci.utils.utils as utils
from deepbci.games.obstacle_avoidance.observation.controller import Controller as OAController

def init_paths(subject_number, trial_number, save_path):
    """Initialize folders and files that OA will use.

    Args:
        subject_number (int): Number of subject for directory/file labeling 

        trial_number (int): Trial number for directory/file labeling.

        save_path (string): abs path to desired save directory.
    """
    paths = {}
    subject = "S" + subject_number
    trial = "trial-" + trial_number

    # Create and save path to subject's trial folder
    subject_trial_dir = join(save_path, subject, trial)
    if not os.path.exists(subject_trial_dir):
        os.makedirs(subject_trial_dir)
    paths["subject_trial_dir"] = subject_trial_dir
    
    # Create and save path for all state information
    state_file = "state-info.csv"
    state_path = join(subject_trial_dir, state_file)
    paths["state_file"] = state_path

    # Create and save path for all collision times
    timing_file = "time-ern.csv"
    timing_path = join(subject_trial_dir, timing_file)
    paths["timing_file"] = timing_path

    # Create and save path for all sync information needed for EEG
    game_info_file =  "game-info.csv"
    game_info_path = join(subject_trial_dir, game_info_file)
    paths["game_info_file"] = game_info_path

    # Create and save path to images captured during the game
    state_img_folder = join(subject_trial_dir, "states")
    if not os.path.exists(state_img_folder):
        os.makedirs(state_img_folder)

    state_imgs_path = join(state_img_folder, 'human-state-images.npy')
    paths["state_imgs_file"] = state_imgs_path

    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--subject-number", default='0',
        help="Pass subject number with leading zero for single digits (Ex: 01 or 11)")
    parser.add_argument("-t", "--trial-number", default='0', help="Trial number")
    parser.add_argument("--seed", default=None, help="Seed number (type int)")
    args = parser.parse_args()
    
    # Set abs data directory save path
    deepbci_path = utils.get_module_root()
    data_dir = "data/obstacle_avoidance/observation"
    save_path = utils.path_to(root_dir=deepbci_path, target=data_dir)
    paths = init_paths(args.subject_number, args.trial_number, save_path)
    
    # Load default values
    kwargs = utils.load_yaml(join(dirname(__file__), "config.yml"))
    
    # Set seed
    if args.seed is not None:
        kwargs['seed'] = int(args.seed)
        
    # Run the game 
    utils.tf_allow_growth()
    oa = OAController(paths=paths, **kwargs)
    oa.run()