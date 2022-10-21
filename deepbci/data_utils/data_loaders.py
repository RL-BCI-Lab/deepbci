import os
from abc import ABC, abstractmethod
from pdb import set_trace
from os.path import join, exists

import pandas as pd
import numpy as np
import mne
from PIL import Image

import deepbci.utils.utils as utils
import deepbci.data_utils as dutils
from deepbci.utils.compress import zstd_decompress
from deepbci.data_utils.build_data.utils import get_nearest_timestamps

def load_data(data_loader, load_method, load_method_kwargs):
    if hasattr(data_loader, load_method):
        bound_load_method = getattr(data_loader, load_method)
    else:
        err = f"The data_loader {data_loader.__class__.__name__} does not contain " \
              f"the load_method {load_method}. Please specify the " \
              f"name of a method that belongs to the data_loader."
        raise AttributeError(err)
    data = bound_load_method(**load_method_kwargs)
    return data 

class _BaseEEGLoader(ABC, object):
    """
        Abstract class for data loaders to implement.
    """
    def __init__(self, abs_dir_path, name, chs, ch_types, fs, event_ids):
        self._abs_dir_path = abs_dir_path
        self._name = name
        self._chs = chs
        self._ch_types = ch_types
        self._fs = fs
        self._event_ids = event_ids
        self._montage = 'standard_1020'
        self._units = 1e-6
        
    @abstractmethod
    def load_to_memory(self):
        pass
        
    @property
    def abs_dir_path(self):
        return self._abs_dir_path

    @property
    def name(self):
        return self._name
    
    @property
    def chs(self):
        return self._chs
    
    @property
    def ch_types(self):
        return self._ch_types
    
    @property
    def fs(self):
        return self._fs
    
    @property
    def event_ids(self):
        return self._event_ids

class _DBCIBaseLoader(_BaseEEGLoader):
    _subject_template = 'S{}'
    _data_file_template = join('filtered', '{}')
    _interval_file_template = 'time-{}'
    _trial_template = 'trial-{}'
    _state_info_file = 'state-info.csv'
    _states_file = join('states', 'state-images.npy')
    _info_columns = ['timestamps', 'labels']
    
    def __init__(self, **kwargs):
        super(_DBCIBaseLoader, self).__init__(**kwargs)

    def load_to_memory(
        self, 
        subjects, 
        trials, 
        data_file, 
        preload_epoch_indexes=None,
        epoch_on_load=None,
        true_fs=False,
        raw_kws=None,
        load_state_info=False,
        load_state_info_kwargs={},
        select_channels=[],
        subject_override={}
    ):
        """
            General loading process for DBCI datasets. 
            
            This class acts as a abstract class and should not be directly called.
           
            Args:
                subject (list): List of ints specifying the subject to load based on the
                    subjects number.
                    
                trials (list): List of ints specifying the trails for a subject to load
                    based on the trial number.
                    
                data_file (str): File name for the data to load from each subject
                
                preload_epoch_indexes (dict | None): Pre-load epoch indices using the specified
                    epoch type. The key represents Epoch types correspond to functions 
                    in data_utils.epoch. The value represents the kwargs for epoching_type.
                    If you wish to pass no kwargs simply pass an empty dict or None. 
                    Example, {load_async_epochs: {}} or {load_async_epochs: None}.
                    Choices for functions are as follows: load_async_epochs, 
                    generate_async_epochs, and generate_sync_epochs.
                    
                epoch_on_load (dict | None): Epoch data as raw data is loaded to save memory.
                    If an empty dictionary or dictionary with mne.Epoch kwargs is provided
                    then epoching will be attempted on the loading of each file. This
                    parameter requires preload_epoch_indexes in order to epoch as it provides
                    the indexes to create epochs at.

                true_fs (bool): Use the true sampling rate calculated from the data.
                
                raw_kws (dict): Kwargs corresponding to the mne.io.RawArray class.
                
                load_state_info (bool): Determines if state info should also be loaded
                    as metadata.
                    
                load_state_info_kwargs (dict): Kwargs for the _load_state_info() method.
                
                subject_override (dict | None): Dictionary where the keys are subjects 
                    numbers and values are arguments to override per individual subject.
                    You can only currently override the 'trials' and 'data_file' args 
                    for each subject.
        """
        def load_epoch_indexes():
            if epoching_type == 'generate_sync_epochs':
                epoch_events = dutils.epoch.generate_sync_epochs(labels=l, 
                                                                 **generate_epoch_kws)
            elif epoching_type == 'generate_async_epochs':
                epoch_events = dutils.epoch.generate_async_epochs(labels=l,
                                                                  eeg_ts=ts,
                                                                  **generate_epoch_kws)
            elif epoching_type == 'load_epochs':
                state_file_path = join(trial_dir_path, 'state-info.csv')
                epoch_events = dutils.epoch.load_epochs(eeg_ts=ts, 
                                                        file_path=state_file_path,
                                                        map_type=self._async_mtype,
                                                        columns=self._info_columns,
                                                        **generate_epoch_kws)
            else:
                raise ValueError(f"Invalid epoching type: received {epoching_type}")
            
            return dutils.mne_ext.generate_events(epoch_events[:, 0], 
                                                  epoch_events[:, 1])   
        if len(select_channels) != 0:
            selected_chs = np.isin(self.chs, select_channels)
            self._chs = np.asarray(self.chs)[selected_chs].tolist()
            self._ch_types = np.asarray(self._ch_types)[selected_chs].tolist()

        # Create mne.info object
        info = mne.create_info(
            ch_names=self.chs, 
            sfreq=self.fs, 
            ch_types=self.ch_types
        )
        
        # Init mne.io.RawArray kwargs
        raw_kws = dict(verbose='WARNING') if raw_kws is None else raw_kws
        
        # Parse information  for generate epochs on load
        if preload_epoch_indexes is not None:
            (epoching_type, generate_epoch_kws), = preload_epoch_indexes.items()
            generate_epoch_kws = {} if generate_epoch_kws is None else generate_epoch_kws
        
        raw_data = []
        for s in subjects:
            override = subject_override.get(s, {})
          
            subject_dir_path = join(self.abs_dir_path,  self._subject_template.format(s))
            
            s_trials = np.array(override.get('trials', trials))

            s_trials.sort()

            for tl in s_trials:
                # Init trail metadata to empty dict
                metadata = {}
                
                # Build trial directory. Each file is located within a trial directory.
                trial_dir_path = join(subject_dir_path,  self._trial_template.format(tl))
                if not exists(trial_dir_path):
                    raise OSError("Trial directory {} was not found".format(trial_dir_path))
                
                # Build file path 
                data_file = override.get('data_file', data_file)
                data_file_path = join(trial_dir_path, self._data_file_template.format(data_file))
                
                if load_state_info:
                     metadata['state_info'] = self._load_state_info(trial_dir_path, **load_state_info_kwargs)
                   
                # Load data
                d, l, ts = load_formated_file(data_file_path, self.chs)

                # Use estimate sampling rate
                if true_fs: 
                    info['sfreq'] = generate_true_fs(ts[0], ts[-1], len(ts))

                # Build RawData
                raw = mne.io.RawArray(d.transpose(1, 0)*self._units, info, **raw_kws)
                # raw.info['sfreq'] = self.fs 
                raw.set_montage(montage=self._montage)
                
                # Update times with DBCI times. So far it doesn't look like there is 
                # another way to update times...oh well, this will do for now.
                raw._times = ts
                
                # Add stim channel and events
                dutils.mne_ext.add_stim_channel(raw, self.fs)
                events = dutils.mne_ext.generate_events(np.arange(len(l)), l)
                raw.add_events(events)
                
                # Add dataset tags 
                tags = dict(dataset=self.name, subject=s, trial=tl)

                # Initialize DataGroup object  
                data = dutils.DataGroup(
                    data=raw, 
                    labels=l,
                    tags=tags,
                    event_ids=self.event_ids
                )
                data.add_metadata(metadata)
                
                # Epoching
                if epoch_on_load is not None:
                    if  preload_epoch_indexes is None:
                        err = "In order to use epoch_on_load you must provide " \
                            "preload_epoch_indexes as well."
                        raise ValueError(err)
                    data.epochs = load_epoch_indexes()
                    dutils.mutators.epochs(data, **epoch_on_load)
                elif preload_epoch_indexes is not None:
                    data.epochs = load_epoch_indexes()

                raw_data.append(data)
                
        return raw_data
    
    def _load_state_info(self, file_path, *, resize=None, clean=True):
        state_info_file = join(file_path, self._state_info_file)
        state_df = pd.read_csv(state_info_file)[self._state_info_cols]
   
        states_file = join(file_path, self._states_file)
        states = self._load_states(states_file)
        
        if resize:
            new_states = []
            for state in states:
                state = Image.fromarray(state)
                state = state.resize(size=resize)
                new_states.append(np.array(state))
            states = np.stack(new_states)
        
        assert len(state_df) == len(states)
        return (state_df, states)
        
    def _load_states(self, file_path, clean=True):
        states = []
        zstd_path = "{}.zst".format(file_path)
        print("Checking if .npy file exists...")
        if os.path.exists(file_path):
            print("Loading .npy file")
            states = np.load(file_path)
        elif os.path.exists(zstd_path):
            print("Attempting to decompress zstd file...")
            zstd_decompress(zstd_path=zstd_path)
            print("Decompressed zstd file")
            if os.path.exists(file_path):
                print("Loading .npy file")
                states = np.load(file_path)
        else:
            raise FileNotFoundError("No state .npy or .npy.zst file detected!") 
            
        if clean: 
            print("Removing {}".format(file_path))
            os.remove(file_path)

        return states        

class BGSObsLoader(_DBCIBaseLoader):
    """
        Data loader for the binary goal search observation dataset.
    """
    def __init__(self):
        module_root = utils.get_module_root()
        rel_dir_path = "data/binary_goal_search/observation"
        
        super_kws = dict(
            abs_dir_path=utils.path_to(module_root, rel_dir_path),
            name='BGSObs',
            chs=['F4', 'F3', 'Fz', 'Cz'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            fs=200,
            event_ids=dict(unk=0, ern=1, crn=2)
        )
        super(BGSObsLoader, self).__init__(**super_kws)
        
        self._async_mtype = 'down'
        self._state_info_cols = ["timestamps", "actions", "terminal"]
        
class BGSIntLoader(_DBCIBaseLoader):
    """
        Data loader for the binary goal search interaction dataset.
    """
    def __init__(self):
        module_root = utils.get_module_root()
        rel_dir_path = "data/binary_goal_search/interaction"
        
        super_kws = dict(
            abs_dir_path=utils.path_to(module_root, rel_dir_path),
            name='BGSInt',
            chs=['F4', 'F3', 'Fz', 'Cz'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            fs=200,
            event_ids=dict(unk=0, ern=1, crn=2)
        )
        super(BGSIntLoader, self).__init__(**super_kws)
        
        self._async_mtype = 'down'
        self._state_info_cols = ["timestamps", "actions", "terminal"]
       
        
class OAObsLoader(_DBCIBaseLoader):
    """
        Data loader for the obstacle avoidance observation dataset.
    """
    def __init__(self):
        module_root = utils.get_module_root()
        rel_dir_path = "data/obstacle_avoidance/observation"
        
        super_kws = dict(
            abs_dir_path=utils.path_to(module_root, rel_dir_path),
            name='OAObs',
            chs=['F4', 'F3', 'Fz', 'Cz'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            fs=200,
            event_ids=dict(unk=0, ern=1)
        )
        super(OAObsLoader, self).__init__(**super_kws)
        
        self._async_mtype = 'balance'
        self._state_info_cols = ["timestamps", "actions", "rewards"]


class OAOutLoader(_DBCIBaseLoader):
    """
        Data loader for the obstacle avoidance outcome dataset.
    """
    def __init__(self):
        module_root = utils.get_module_root()
        rel_dir_path = "data/obstacle_avoidance/outcome"
        
        super_kws = dict(
            abs_dir_path=utils.path_to(module_root, rel_dir_path),
            name='OAOut',
            chs=['F4', 'F3', 'Fz', 'Cz'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            fs=200,
            event_ids=dict(unk=0, ern=1)
        )
        super(OAOutLoader, self).__init__(**super_kws)
        
        self._async_mtype = 'balance'
        self._state_info_cols = ["timestamps", "actions", "rewards"]

def load_formated_file(path, channels):
    df = pd.read_csv(path, index_col=None)
    return df.loc[:, channels].values, df['labels'].values, df['timestamps'].values

def generate_true_fs(start, end, samples):
    return samples / (end - start)
