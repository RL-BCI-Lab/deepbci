import pdb
import os
import math 
import datetime
from pdb import set_trace
from math import ceil

import numpy as np
import pandas as pd

from deepbci.utils import utils as core

def find_nearest(data, target, map_type='balance', down_thresh=0):
    """ Find the nearest value in a given array 

        Args:
            timestamps (ndarray): 1D list of numerical values
            
            target (float): Real number value that you would like to match as 
                close as possible to a number in the given data array.
            
        Returns:
            Returns the value from the data closest to the target and its index.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if map_type.lower() == 'balance':
        diffs = np.abs(data-target)
        idx = diffs.argmin()
    elif map_type.lower() == 'down':
        diffs = data - target
        # Will throw argmax error if no values are less than threshold
        if len(diffs[diffs <= down_thresh]) == 0:
            raise ValueError("No timestamps were found below the mapping threshold.")
        idx = diffs[diffs <= down_thresh].argmax()
    else:
        raise Exception("Invalid map_type error type")
    # print("{} -> {}".format(target, data[idx]))
    return data[idx], idx

def get_nearest_timestamps(base_timestamps, target_timestamps, **kwargs):
    """ Maps target timestamps to base timestmaps"""
    mapped_ts = []
    mapped_idx = []
    offset = 0
    # total_error = 0

    # NOTE: Added so 1D arrays will not be squeezed on axis 1 which does not exist.
    if len(target_timestamps.shape) > 2:
       target_timestamps = np.squeeze(target_timestamps, axis=1)

    for ts in target_timestamps:
        map_ts, map_idx = find_nearest(data=base_timestamps[offset:], 
                                       target=ts, 
                                       **kwargs)
        map_ts = map_ts[0] if isinstance(map_ts, np.ndarray) else map_ts
        
        offset += map_idx+1
        
        mapped_ts.append(map_ts)
        mapped_idx.append(offset-1)
        
    #     total_error += np.abs(ts - map_ts)
    #     print("base timestamp used: {}/{}".format(mapped_idx[-1], len(base_timestamps)))
    #     print("OG: {} -> NEW: {}".format(ts, map_ts))
    #     print("Error: {:.5f}\n".format(np.abs(ts - map_ts)))
        
    # print("total error: {}".format(total_error))
    if len(np.unique(np.hstack(mapped_ts))) != len(np.hstack(mapped_ts)):
        raise Exception("Potential duplicate timesteps detected!")
    
    return np.hstack(mapped_ts), np.hstack(mapped_idx)

def times_to_sec(times):
    """ Convert an array of time stamps to time in seconds

        Args:
            times (list): List of time stamp strings foramtted as hh:mm:ss.msms.
        
        Returns:
            Ruturns a list of float time stamps in seconds.
    """
    if not isinstance(times, np.ndarray):
        raise ValueError("Invalid type {}. Time must be of type ndarray".format(
            type(times)))

    times_in_sec = np.zeros(times.shape)
    for i, time in enumerate(times):
        times_in_sec[i] = time_to_sec(time)

    return times_in_sec

def time_to_sec(time):
    """ Covert time stamp of format hh:mm:ss.msms to seconds.

        Args:
            time (str): Time stamp with formatted as hh:mm:ss.msms.

        Returns:
            Float time stamp in seconds
    """
    time = time.replace(" ", "")
    hms = [float(i) for i in time.split(":")]
    if len(hms) < 3:
        missing = 3 - len(hms)
        for i in range(missing):
            hms.append(0)
    seconds = hms[0]*3600 + hms[1]*60 + hms[2]

    # Datatime is significantly slower
    # dt = datetime.datetime.strptime(time, '%H:%M:%S.%f')
    # assert seconds == (dt-datetime.datetime(1900,1,1)).total_seconds()
    return seconds

def sec_to_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

def lerp(x0, x1, t):
    return (1- t) * x0 + t * x1

def _generate_timestamps(start, end, missing):
    times = []

    for i in range(1, missing+1):
        t = (i/(missing+1))
        seconds = lerp(time_to_sec(start), time_to_sec(end), t)
        times.append(str(datetime.timedelta(seconds=seconds))[:-3])

    return times

def _generate_data(start, end, missing):
    syn_data = []
    for i in range(1, missing+1):
        t = (i/(missing+1)) 
        data = np.round(lerp(start, end, t), 2)
        syn_data.append(data)
    return syn_data

def fill(shape, fill_with=0):
    empty = np.empty(shape)
    empty.fill(fill_with)
    return empty

class OpenBCISampleFill:
    """ Sample filling for OpenBCI Formated Data"""
    def __init__(self, data, utils, fs):
        self.data = data
        self.utils = utils
        self.iter = utils[:,0]
        self.fs = fs
        self.minfo = []

    def __call__(self, fill_limit=None):
        fill_limit = self.fs if fill_limit is None else fill_limit

        midx, miter_pos, miters = self.find_missing()
        minfo = self.synthesize(midx, miter_pos, miters, fill_limit)
        self.insert_synthetic(minfo)

        self._check_monotonicity()
        
        return self.data, self.utils, minfo

    def find_missing(self):
        diffs = abs(np.diff(self.iter))
        potential_midx = np.where((diffs != 1) & (diffs != self.fs))[0]

        midx = []
        miters = []
        miter_pos = []
        for i, idx in enumerate(potential_midx):
            iters = []
            s = self.iter[idx]
            e = self.iter[idx+1]

            if s > e:
                iters += range(s+1, self.fs+1)
                iters += range(0, e)
            elif s < e:
                iters += range(s+1, e)
  
            midx.append(idx)
            miters.append(iters)
            miter_pos.append((s, e))
                
        # Check start for missing samples (will not fill)
        if self.iter[0] != 0:
            print(core.bordered(f"WARNING: Not filling samples before index {self.iter[0]}!"))
            # miter_start = np.hstack([[0], miter_start])
            # miter_end = np.hstack([self.iter[0], miter_end])
        
        # Check end for missing samples (will not fill)
        if self.iter[-1] != self.fs:
            print(core.bordered(f"WARNING: Not filling samples before index {self.iter[-1]}!"))
            # miter_start = np.hstack([self.iter[-1], miter_start])
            # miter_end = np.hstack([self.fs, miter_end])

        return midx, miter_pos, miters
    
    def synthesize(self,midx, miter_pos, miters, fill_limit):       
        offset = 0
        minfo = {}
        for idx, (s, e), iters in zip(midx, miter_pos, miters):
            
            if len(iters) > fill_limit:
                # data = _fill(shape=(len(iters), self.data.shape[-1]))
                warn = f"WARNING: Not interpolating {len(iters)} samples at index " \
                       f"{idx} with start iter {s} and end iter {e}!"
                print(core.bordered(warn))
                continue
            else:
                data = _generate_data(start=self.data[idx], 
                                    end=self.data[idx+1], 
                                    missing=len(iters))
                
            times = _generate_timestamps(start=self.utils[idx, 1], 
                                        end=self.utils[idx+1, 1], 
                                        missing=len(iters))
            
            utils = np.empty((len(iters), 2), dtype=np.object)
            utils[:, 0], utils[:, 1] = iters, times
            start_idx = idx + offset
            end_idx = start_idx + len(iters)
          
            minfo[idx] = {
                'start': idx,
                'start_offset': start_idx,
                'end_offset': end_idx, 
                'siter': s,
                'eiter': e,
                'data': data,
                'utils': utils
            }
            offset += len(iters)
        return minfo

    def insert_synthetic(self, minfo):
        for _, info in minfo.items():
            
            # Add one to index as NumPy inserts BEFORE passed index not after
            self.data = np.insert(self.data, info['start_offset']+1, info['data'], axis=0)
            self.utils = np.insert(self.utils, info['start_offset']+1, info['utils'], axis=0)
            # Update iters with newly inserted iters.
            self.iter = self.utils[:,0]
            
            # print(self.data[info['start_offset']-1:info['end_offset']+1])
    
    def _check_monotonicity(self):
        """ Check to make sure all timestamps are monotonically increasing. """
        ts = times_to_sec(self.utils[:, 1])
        diff = np.diff(ts)
 
        if len(diff[diff < 0]) > 0:
            set_trace()
            raise ValueError("Timestamps are not monotonically increasing!")
    
