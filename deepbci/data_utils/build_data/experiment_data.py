from distutils.command.build import build
import sys
import os
from math import ceil, floor
from os.path import join
from pdb import set_trace
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.signal import resample

import deepbci
import deepbci.utils.utils as utils
from deepbci.data_utils.exp_loaders import load_state_data, load_task_events, load_task_info
from deepbci.data_utils.build_data.filters import Filters
from deepbci.data_utils.build_data.utils import OpenBCISampleFill
from deepbci.data_utils.build_data.utils import find_nearest, times_to_sec, time_to_sec, sec_to_time
from deepbci.data_utils.build_data.utils import get_nearest_timestamps

class _Info():
    def __repr__(self):
        return type(self).__name__
        
    def print_info(self, logger=print):
        logger(self)
        for key, value in self.__dict__.items():
            logger(f"\t{key}: {value}")
            
class TaskInfo(_Info):
    clbr_start: str
    clbr_start_norm: float
    clbr_start_secs: float
    start: str
    start_norm: float
    start_secs: float
    end: str
    end_secs: float
    dur_with_clbr: float
    dur: float

class EEGInfo(_Info):
    start: str
    start_secs: float
    end: str
    end_secs: float
    dur: float
    fs: float

class ExperimentData():
    """ Stores all DBCI experiment data including both EEG and task data.

        Attributes:

        Args:
            fill (bool): Determine if filling of dropped samples should be done.

            fill_limit (int): If fill is True then this determines the maximum number
                of samples to attempt to synthesize. The larger the consecutive gap,
                the more inaccurate the filling of synthetic samples.  

            build_timestamps (bool): If true will build custom time stamps instead of
                reusing timestamps in EEG file.
    """
    def __init__(
        self, 
        eeg_loader, 
        eeg_path, 
        state_info_path, 
        task_info_path, 
        event_paths,
        fill=False, 
        fill_limit=None,
        before_padding=0,
        after_padding=0,
        build_timestamps=False,
    ):

        self.eeg_loader = eeg_loader
        self._eeg_path = eeg_path
        self._state_info_path = state_info_path
        self._task_info_path = task_info_path
        self._event_paths = event_paths
        self.fill = fill
        self.fill_limit = fill_limit
        self.before_padding = before_padding
        self.after_padding = after_padding
        self.build_timestamps = build_timestamps
        
        self.true_sampling_rate = None
        self._sync_synthetic = False
        self._eeg = None 
        self._eeg_ts = None
        self._eeg_itr = None
        self._eeg_labels = None
        self._task_state = None
        self._task_event_ts = {}
        self._task_ts = None
        self._epochs = None
        self.fill_info = None

        self._task_info = TaskInfo()
        self._eeg_info = EEGInfo()
            
    def load(self):
        """ Loads DBCI specific experiment data data"""     
        # Load eeg data 
        eeg_df, time_df = self.eeg_loader.load(self._eeg_path)
        self._eeg = eeg_df.values
        self._eeg_itr, self._eeg_ts = time_df.iloc[:, 0].values, time_df.iloc[:, 1].values
        
        # Load task state information
        self._task_state = load_state_data(self._state_info_path).values
        
        # Load signal estimated timestmaps
        for label, path in self._event_paths.items():  
            self._task_event_ts[label] = load_task_events(path).values
            
        # Load all task recorded timestamps
        self._task_ts = self._task_state[:, 0]
        
        # Load task start and end timestamp information
        task_info = load_task_info(self._task_info_path)
        self._set_task_info(task_info)
        self._set_eeg_info()

        # Attempt to interpolate any missing samples
        if self.fill:
            self._fill_samples(self.fill_limit)

        if self.build_timestamps: 
            self._build_synthetic_timestamps()

    def _set_task_info(self, task_info):
        self._task_info.clbr_start = sec_to_time(task_info['before_start'])
        self._task_info.clbr_start_secs = task_info['before_start']
        self._task_info.clbr_start_norm = 0
        self._task_info.start = sec_to_time(task_info['start'])
        self._task_info.start_norm = task_info['start'] - task_info['before_start']
        self._task_info.start_secs = task_info['start']
        self._task_info.end = sec_to_time(task_info['end'])
        self._task_info.end_secs = task_info['end']
        self._task_info.dur_with_clbr = task_info['end'] - task_info['before_start']
        self._task_info.dur = task_info['end'] - task_info['start']
    
    def _set_eeg_info(self):
        self._eeg_info.start = self._eeg_ts[0]
        self._eeg_info.start_secs = time_to_sec(self._eeg_ts[0])
        self._eeg_info.end = self._eeg_ts[-1]
        self._eeg_info.end_secs = time_to_sec(self._eeg_ts[-1])
        self._eeg_info.dur = time_to_sec(self._eeg_ts[-1]) - time_to_sec(self._eeg_ts[0])
        self._eeg_info.approx_dur = len(self._eeg_ts) / self.eeg_loader.fs
        self._eeg_info.fs = len(self._eeg_ts) / self._eeg_info.dur

    def _fill_samples(self, fill_limit):
        eeg_utils = np.stack([self._eeg_itr, self._eeg_ts], axis=1)
        fill = OpenBCISampleFill(self._eeg, eeg_utils, self.eeg_loader.fs)
        self._eeg, eeg_utils, self.fill_info = fill(fill_limit)
        self._eeg_itr, self._eeg_ts = eeg_utils[:, 0], eeg_utils[:, 1]

    def _build_synthetic_timestamps(self):
        # Use custom timestamps for syncing
        self._sync_synthetic = True
        # Create timestamps based on "actual" sampling rate
        # as sampling rates vary slightly everytime a new recording is done
        self._eeg_ts = np.arange(len(self._eeg_ts)) / self._eeg_info.fs
        
    def butter_filter(self, ftype, **kwargs):
        ftype = getattr(Filters, ftype)
        self._eeg =  ftype(self._eeg, fs=self.eeg_loader.fs, axis=0, **kwargs)
        
    def _run_filter(self, filter_func, **kwargs):
        clean_data = []
        for i in range(self._eeg.shape[1]):
            clean_data.append(filter_func(data=self._eeg[:, i],
                                          fs=self.eeg_loader.fs,
                                          **kwargs))
        return clean_data

    def sync_data(self):
        if self._sync_synthetic:
            self._sync_via_synthetic_timestamps()
        else:
            self._sync_via_timestamps()
            
    def _sync_via_synthetic_timestamps(self):
        """ Sync synthetic EEG timestamps with task start and end timestamps.
            
            Padding is currently based on timestamps. This will lead to an inconsistent
            number of samples before and after each trial. To have a consistent number 
            of samples, padding needs to be based on samples using sampling rate - similar
            how fake timestamps are built.

        """
       
        start_norm = self._task_info.start_secs - self._eeg_info.start_secs
        _, start_idx = find_nearest(self._eeg_ts, start_norm, map_type='balance')
        
        start_norm_p = (self._task_info.start_secs-self.before_padding) - self._eeg_info.start_secs
        _, start_idx_p = find_nearest(self._eeg_ts, start_norm_p, map_type='balance')
        
        end_norm_p = (self._task_info.end_secs+self.after_padding) - self._eeg_info.start_secs
        _, end_idx_p = find_nearest(self._eeg_ts, end_norm_p, map_type='balance')

        # Normalize data from correct starting point
        self._eeg_ts -= self._eeg_ts[start_idx]
        # Set synced eeg data and timestamps
        self._eeg =  self._eeg[start_idx_p:end_idx_p]           
        self._eeg_ts = self._eeg_ts[start_idx_p:end_idx_p]

    def _sync_via_timestamps(self):
        """ Sync EEG timestamps with task start and end timestamps.
            
            Padding is currently based on timestamps. This will lead to an inconsistent
            number of samples before and after each trial. To have a consistent number 
            of samples, padding needs to be based on samples using sampling rate - similar
            how fake timestamps are built.

        """
        times = times_to_sec(self._eeg_ts)
        # norm_times = np.round(times - self._task_info['before_start'], 3)
        
        # Get True start time
        _, start_idx = find_nearest(times, self._task_info.start_secs, map_type='balance')

        _, start_idx_p = find_nearest(times, self._task_info.start_secs-self.before_padding, map_type='balance')

        _, end_idx_p = find_nearest(times, self._task_info.end_secs+self.after_padding, map_type='balance')

        # Normalize data from correct starting point
        times -= times[start_idx]
        # Set synced eeg data and timestamps
        self._eeg =  self._eeg[start_idx_p:end_idx_p]
        self._eeg_ts = times[start_idx_p:end_idx_p] 
    
    def build_labels(self, nearest_kwargs=None):
        nearest_kwargs = {} if nearest_kwargs is None else nearest_kwargs
        
        self._eeg_labels = np.zeros((len(self._eeg_ts), 1), dtype=int)
        
        for signal_label, signal_ts in self._task_event_ts.items():
            # print("Signal label: {} Signal count: {}".format(signal_label, len(signal_ts)))
            ts, indices = get_nearest_timestamps(base_timestamps=self._eeg_ts, 
                                                 target_timestamps=signal_ts, 
                                                 **nearest_kwargs)
            self._eeg_labels[indices] = signal_label

    @property
    def eeg_info(self):
        return self._eeg_info

    @property 
    def eeg(self):
        return self._eeg
    
    @property
    def eeg_ts(self):
        return self._eeg_ts
    
    @property
    def eeg_iters(self):
        return self._eeg_itr

    @property
    def eeg_labels(self):
        return self._eeg_labels
    
    @property
    def task_state(self):
        return self._task_state
    
    @property
    def task_event_ts(self):
        return self._task_event_ts
    
    @property
    def task_ts(self):
        return self._task_ts
    
    @property
    def task_info(self):
        return self._task_info

    @property
    def epochs(self):
        return self._epochs
    