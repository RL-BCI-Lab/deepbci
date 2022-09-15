import numpy as np
import pandas as pd
import os
import re
import argparse
from os import listdir
from os.path import join, isfile
from pdb import set_trace
from pathlib import Path
from shutil import copyfile

"""
Place this script in the directory where the OpenBCI raw data directories are
stored after recording. Change the  DBCI_DATA_REL to a path within your home
directory (a path will be made automatically to the relative path of the
specified directory). Change DIR_PREFIX to determine which directories should
be looked at. Target directories must have the following format:
    <dir_prefix><subject>-<task>-<subtask>-<trial>
"""

# Arg parsing 
parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', 
                    help="dry run script without moving files")
args = parser.parse_args()
DRYRUN = args.dry_run

# Get working directory
WORKING_DIR = os.getcwd()
# Find the directories where this script is running that start with the following
DIR_PREFIX = 'OpenBCISession_'
# Extract HOME path from OS
HOME = str(Path.home())
# Path to directory formated like dbci data folder structure
DBCI_DATA_REL = ['space', 'deep-bci', 'deepbci', 'data']
# Abbreviated task to actual task name
TASKS = {
    'oa': 'obstacle_avoidance',
    'bgs': 'binary_goal_search'
}
# Abbreviated sub-task to actual sub-task name
SUBTASKS = {
    'obs': 'observation',
    'int': 'interaction',
    'out': 'outcome'
}
TRIAL_NAME = 'trial'
SUBJECT_TEMPLATE = 'S[0-9]?[0-9]'

def update_task(dir_):
    for key, value in TASKS.items():
        if key in dir_.lower():
            return value

def update_subtask(dir_):
    for key, value in SUBTASKS.items():
        if key in dir_.lower():
            return value
        
def path_to(root_dir, target, topdown=True, search_method=str.startswith):
    """ Search directory for a target file/folder and construct an absolute 
        path to target.
    """
    if not isinstance(target, list):
        target = target.split(os.path.sep)
        
    for root, folders, files in os.walk(root_dir, topdown=topdown):
        found = folders + files
        for name in found:
            abs_path = os.path.join(root, name)
            path = abs_path.split(os.path.sep)
            if search_method(name, target[-1]):
                potential_path = target[:]
                potential_path[-1] = name
                if all(elm in path for elm in potential_path):
                    return os.path.normpath(abs_path)
            
    raise FileNotFoundError("Target {} was not found in {}".format(join(*target), root_dir))

def build_name(dir_):
    # Remove spaces from found data dir
    dir_.replace(' ', '')
    split_dir = dir_.split('-')
    # Extract task 
    task = update_task(dir_)
    # Extract sub-task
    subtask = update_subtask(dir_)
    # new_dir = new_dir.split('-')
    subject = re.search(SUBJECT_TEMPLATE, dir_).group(0).upper()
    # Remove 0 from S0x
    if subject[1] == '0':
        subject = subject[0] + subject[-1:]
    trial = TRIAL_NAME + '-' + split_dir[-1].replace('t', '')
    # Remove 0 from trial-0x
    if trial[-2] == '0':
        trial = trial[:-2] + trial[-1]

    return join(task, subtask, subject, trial)

def move_dir(original, new):
    copyfile(original, new)

def get_file_from_dir(target_dir):
    found_files = []
    for f in listdir(target_dir):
        found = join(target_dir, f)
        if isfile(found):
            found_files.append(found)
    if len(found_files) > 1:
        raise ValueError("Too many files found at {}".format(target_dir))
    return found_files[0]

def clean_openbci_file(openbci_file):
    # Does NOT load comment lines!
    df = pd.read_csv(openbci_file, comment='%', header=None)
    df = get_uniques(df)
    df = remove_hanging_samples(df)
    return df

def get_uniques(df):
    unique_values = len(df.drop_duplicates())
    if len(df) != unique_values:
        print("Duplicates detected:") 
        print("\tBase: {}".format(len(df)))
        print("\tUnique: {}".format(unique_values))
        if unique_values != (len(df) / 2):
            err = "Uneven number of unique values detected after pruning duplicates"
            raise Exception(err)
        return df.drop_duplicates()
    return df

def remove_hanging_samples(df):
    last_idx = df.iloc[-1, 0]
    if df.iloc[:-last_idx-1, 0].iloc[-1] != 200:
        raise Exception("After removing haning samples the next index was not 200")
    return df.iloc[:-last_idx-1]
 
def save_dataframe(df, save_path):
    df.to_csv(save_path, header=None, index=False)
    
if __name__ == '__main__':
    string_equality = lambda x, y: x == y
    dbci_data_abs = path_to(HOME, join(*DBCI_DATA_REL), search_method=string_equality)
    directories = os.listdir(WORKING_DIR)
    original = []
    for dir_ in directories:
        if str.startswith(dir_, DIR_PREFIX):
            print("\nCLEANING AND MOVING: {}".format(dir_))
            found_dir_rel = dir_.replace(DIR_PREFIX, '')
            found_dir_abs = join(WORKING_DIR, dir_)
            file_in_dir_abs = get_file_from_dir(found_dir_abs)
            file_in_dir_name = os.path.split(file_in_dir_abs)[-1]
            move_to_dir_rel = build_name(found_dir_rel)
            move_to_dir_abs = join(dbci_data_abs, move_to_dir_rel)
            
            # print(move_to_dir_abs)
            # print(found_dir_abs)
            df = clean_openbci_file(file_in_dir_abs)

            if os.path.exists(move_to_dir_abs):
                new_file_abs =  join(move_to_dir_abs, file_in_dir_name)
                print("{} -> {}".format(file_in_dir_abs, new_file_abs))
                # move_dir(file_in_dir_abs, new_file_abs)
                if not DRYRUN:
                    save_dataframe(df, new_file_abs)

            else:
                error_msg = "New target path can not be found: {}"
                raise FileNotFoundError(error_msg.format(move_to_dir_abs))
