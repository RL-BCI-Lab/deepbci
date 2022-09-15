from multiprocessing.sharedctypes import Value
from pdb import set_trace

import pandas as pd
import numpy as np

class EEGLoader():
    def load(self):
        raise NotImplemented

class OpenBCILoader(EEGLoader):
    _valid_versions = [4, 5]
    _valid_boards = ['ganglion', 'cryton']

    def __init__(self, board, fs, version=5, split=True):
        self.board = board
        self.fs = fs
        self.version = version
        self.split = split

        if self.board not in self._valid_boards:
          raise ValueError(f"`board` must be set equal to one of the following {self._valid_boards}, "
                           f"received {self.board}")

        if self.version not in self._valid_versions:
            raise ValueError(f"`version` must be set equal to one of the following {self._valid_versions}, "
                            f"received {self.version}")

    def load(self, path, ):
        if self.version == 4:
            eeg = self._load_openbci_v4(path)
        elif self.version == 5:
            eeg = self._load_openbci_v5(path)

        return eeg

    def _load_openbci_data(self, path, **kwargs):
        eeg_df = pd.read_csv(path, **kwargs)
        return eeg_df

    def _load_openbci_v4(self, path):
        df = self._load_openbci_data(path, header=None, index_col=None)
        df = self._get_uniques(df)
        df = self._remove_hanging_samples(df)

        # Separate EEG metadata and channel data
        # Column 0: sample rate iterator (max value = sample rate)
        # Column 1-4: Active EEG channels
        # Columns 5-7: Empty channels
        # Column 8-9: Timestamps (when a data sample was received)
        if self.split:
            time_df = df.iloc[:,[0,8]]
            try:
                # New BCI text format contains 9 rows.
                eeg_df = df.iloc[:, [1,2,3,4]]
            except KeyError:
                raise Exception('Data format does not match V4. Check to see if a older version' 
                                ' of the OpenBCI GUI was used to record the data.')
            return eeg_df, time_df

        return df

    def _load_openbci_v5(self, path):
        df = self._load_openbci_data(path, comment='%', sep=', ', engine='python')
        df['Sample Index'] = self._v5_to_v4_index(df['Sample Index'].values)
        df = self._get_uniques(df)
        df = self._remove_hanging_samples(df)
        # By default, ts is formatted as 2022-07-11 13:15:26.050.
        # We only want the 2nd element after being split (actual time not date).
        ts_split_func = lambda ts: ts.split(' ')[1]
        df['Timestamp (Formatted)'] = df['Timestamp (Formatted)'].apply(ts_split_func)


        if self.split:
            time_df = df.loc[:, ['Sample Index', 'Timestamp (Formatted)']]
            if self.board.lower() == 'ganglion':
                eeg_df = df.iloc[:, [1,2,3,4]]
            elif self.board.lower() == 'cyton':
                raise NotImplemented
            return eeg_df, time_df
        
        return df

    def _get_uniques(self, df):
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

    def _remove_hanging_samples(self, df):
        last_idx = df.iloc[-1, 0]
        if df.iloc[:-last_idx-1, 0].iloc[-1] != self.fs:
            raise Exception("After removing hanging samples the next index was not 200")
        return df.iloc[:-last_idx-1]

    def _v5_to_v4_index(self, index):
        new_idx = []
        prev_idx = None
        half_fs = int(self.fs / 2)
        
        odds = np.insert(np.arange(1, self.fs, 2), 0, 0)
        evens = np.insert(np.arange(2, self.fs+1, 2), 0, 0)
        key = np.arange(0, half_fs+1)
        index_map = [
            dict(zip(key, odds)),
            dict(zip(key, evens)),
        ]

        for i, idx in enumerate(index):
            if idx == prev_idx:
                use_map = 1
            elif prev_idx == None or idx == 0:
                use_map = 0
            elif abs(idx - prev_idx) != 1:
                use_map = 0
                # does not include the current mapped value so 1 must be subtracted
                diff = index_map[use_map][idx] - new_idx[i-1] - 1
                print(f"Dropped samples: {diff}")
                print(f"\tIndexes: {i-1}-{i}")
                print(f"\tValues: {prev_idx}-{idx}")
                print(f"\tMapped values: [{new_idx[i-1]}, {index_map[use_map][idx]})")
            else:
                use_map = 0
                
            new_idx.append(index_map[use_map][idx])
            prev_idx = idx
        return np.array(new_idx)

def load_state_data(path):
    state_df = pd.read_csv(path, header=0, index_col=None)
    
    return state_df

def load_task_events(path):
    state_df = pd.read_csv(path, header=None, index_col=None)
    
    return state_df

def load_task_info(path):
    """
        Load task/game timing info for syncing to EEG.
        
        Format:
            Column 0: Game duration in seconds (only of game note countdown)
            Column 1: Game count down starts time stamp in seconds
            Column 2: Game start time stamp in seconds
            Column 3: Total game duration (countdown to game end)time stamp in seconds
    """
    task_info = {}
    task_info_df = pd.read_csv(path, header=None, index_col=None)
    task_info['task_duration'] = round(float(task_info_df.values[0]), 3)
    task_info['before_start'] = round(float(task_info_df.values[1]), 3)
    task_info['start'] = round(float(task_info_df.values[2]), 3)
    task_info['end'] = round(float(task_info_df.values[3]), 3)
    
    return task_info