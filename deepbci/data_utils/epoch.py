from os.path import join, dirname
from pdb import set_trace

import pandas as pd
import numpy as np

from deepbci.data_utils.build_data.utils import get_nearest_timestamps

def generate_sync_epochs(labels, use_events=None):
    """
        Genrerate an synced epochs array which contains epoch indices and labels.
        
        Sync here indicates that we can directly grab the index at which a given
        event_id occurred. Assumes the default label is 0, thus 0 will be ignored unless 
        specified with other events in use_events. It should be noted that using 0 as an 
        epoch will create an epoch at every point labeled 0. If 0 is used as the default
        placeholder event then this can create thousands of epochs which will slow
        down mne.Epochs in the future and require more memory.
        
        labels (np.ndarray): 1D array of labels
                   
        use_events (list | np.ndarray): Events to be used when creating the epoch labels.
    """
    # If use_events is not provide then all non-zero labels are used
    if use_events is None:
        use_events = np.unique(labels)
        use_events = use_events[use_events != 0]
            
    epoch_idx = []
    for e in use_events:
        e_idx = np.where(labels==e)[0]
        epoch_idx.append(e_idx)
    epoch_idx = np.sort(np.hstack(epoch_idx))

    return np.stack([epoch_idx, labels[epoch_idx]], axis=1).astype(int)

def generate_async_epochs(labels, 
                          eeg_ts, 
                          step_size,
                          start_idx=None, 
                          end_idx=None, 
                          use_events=None,
                          label_boundary=None,
                          map_type='down'):
    """
        Genereate async epochs which EEG data and labels will be mapped to.
        
        Assumes the default label is 0, thus 0 will be ignored unless 
        specified with other events in use_events. It should be noted that using 0 as an 
        epoch will create an epoch at every point labeled 0. If 0 is used as the default
        placeholder event then this can create thousands of epochs which will 
        greatly slow down mne.Epochs in the future and require extensive memory.
        
        Args:
            labels (np.ndarray): 1D array of labels for eeg_ts
            
            eeg_ts (np.ndarray): EEG timestamps for each data sample.
            
            step_size (int): Step size between each timestamp in milliseconds (ms)
            
            start_idx (int): Start index for eeg_ts to use when creating async epochs 
            
            end_idx (int): End index for eeg_ts to use when creating async epochs 
            
            use_events (list | np.ndarray): Events to be used when creating the labels
                for the aysnc timstamps (i.e. epochs).

            label_boundary (list | np.ndarray): A lower and upper bound given in
                milliseconds which determines if epochs close to a given label
                should also bee given the same label.

            map_type (str): Corresponds to the type of mapping using by the 
                deepbci.data_utils.build_data.utils.get_nearest_timestamps() function.
                Mapping is used to determine how EEG timestamps are be mapped to 
                the async timestamps. For example, passing 'balance' maps EEG
                timestamp to an async timestamp where the difference between the two is
                the smallest (smallest error). Meanwhile, 'down' always maps the EEG 
                timestamp to the closest async timestamp that came before it (no mapping 
                to future timestamps that have not occurred yet, introduces larger error).
    """
    # Set start_idx to timestamp closest to 0 by default
    if start_idx is None:
        start_idx = (np.abs(eeg_ts)).argmin()

    # Set end_idx to the last timestamps
    if end_idx is None:
        end_idx = -1
    
    # Use all unique labels except for 0 by default 
    if use_events is None:
        use_events = np.unique(labels)
        use_events = use_events[use_events != 0] 
        
    async_ts = np.arange(eeg_ts[start_idx], eeg_ts[end_idx], step_size / 1000)
    
    epoch_labels = np.zeros((len(async_ts)), dtype=int)

    # Create labels by mapping the EEG timestamp to the closest async timestamps
    for e in use_events:
        e_idx = np.where(labels==e)[0]
        label_async_ts, label_async_idx = get_nearest_timestamps(base_timestamps=async_ts,
                                              target_timestamps=eeg_ts[e_idx],
                                              map_type=map_type)

        # WARNING: If boundaries are too large and multiple events are used
        # events can override a previous event's labels due to overlap! 
        if label_boundary is not None:
            label_async_idx = _generate_label_boundary(label_boundary,
                                                       epoch_labels,
                                                       label_async_idx, 
                                                       step_size)

        epoch_labels[label_async_idx] = e

    # Get epoch indices by mapping the async timestamps to the closest EEG timestamps 
    epoch_ts, epoch_indices = get_nearest_timestamps(base_timestamps=eeg_ts,
                                                     target_timestamps=async_ts,
                                                     map_type='balance')      

    return np.stack([epoch_indices, epoch_labels], axis=1)

def _generate_label_boundary(label_boundary, epoch_labels, label_async_idx, step_size):
    """ Generates label boundaries based on epoch indices
    
        Args:
            label_boundary (list | np.ndarray): A lower and upper bound given in
                milliseconds which determines if epochs close to a given label
                should also bee given the same label.

            epoch_labels (np.ndarray): NumPy array of labels for epochs.

            label_async_idx (np.ndarray): NumPy array containing the indices
                for a certain class's labels in epoch_labels.

            step_size (int): Step size between each epoch in milliseconds (ms).
    """
    if not isinstance(label_boundary, np.ndarray):
        label_boundary = np.array(label_boundary)

    if len(label_boundary) != 2:
        err = "The label_boundary length must be equal to 2."
        raise ValueError(err) 

    if not all([i == 0 for i in label_boundary%step_size]):
        err = "The passed label_boundary bounds are not divisible by the step_size."
        raise ValueError(err)
    
    if label_boundary[0] > 0:
        err = "The lower boundary for label_boundary can not be greater than 0."
        raise ValueError(err)

    lower_bounds = label_async_idx + (label_boundary[0]//step_size)
    # Set limit on lower bound index
    outofbounds = np.where(lower_bounds < 0)[0]
    if len(outofbounds) != 0:
        lower_bounds[outofbounds] = 0
    
    upper_bounds = label_async_idx + (label_boundary[1]//step_size)
    # Set limit on upper bound index
    outofbounds = np.where(upper_bounds > len(epoch_labels))[0]
    if len(outofbounds) != 0:
        upper_bounds[outofbounds] = len(epoch_labels)-1

    # Generate all indices between lower and upper bounds and making sure
    # the original index is included!
    label_boundary_indices = []
    for l, u, og_idx in zip(lower_bounds, upper_bounds, label_async_idx):
            lbi = np.arange(l, u+1)
            assert np.any(np.isin(lbi, og_idx))
            label_boundary_indices.append(lbi)
    label_boundary_indices = np.hstack(label_boundary_indices)
    assert len(np.unique(label_boundary_indices)) == len(label_boundary_indices)
    
    return label_boundary_indices

def load_epochs(eeg_ts, columns, file_path, map_type='balance', use_events=None):
    """
        Load epochs via a file which contains epochs timestamps and labels. 
        
        epoch file should contain epoch start timestamp and corresponding label. 
        The timestamp column should be specified in by the first element in the columns 
        parameter. This can be a column index or column header (str). Likewise, the 
        corresponding label column should be given as the second element in the columns 
        parameter. If epoch timestamps do not exactly match EEG timestamps then a mapping 
        to the nearest timestamp is done (mapping decided via mtype kwarg). 
        
        Args:
        
            eeg_ts (np.ndarray): EEG timestamps for each data sample.
            
            columns (list): A list containing all str or all int. The first element 
                corresponds to the location of the timestamps and the second element 
                corresponds to the location of the labels.
            
            file_path (str): Path to file contaiing async timestamps and labels.
            
            map_type (str): Corresponds to the type of mapping using by the 
                deepbci.data_utils.build_data.utils.get_nearest_timestamps() function.
                Mapping is used to determine how loaded timestamps are be mapped to 
                the passed EEG timestmaps. For example, passing 'balance' maps  the given
                timestamp to an EEG timestamp where the difference between the two is
                the smallest (smallest error). Meanwhile, 'down' alwaysmaps the given 
                timestamp to the closest EEG timestamp that came before it (no mapping 
                to future timestamps that have not occurred yet, can introduce larger error).
    
    """
    df = pd.read_csv(file_path, header=0, index_col=None)
    
    if all([isinstance(c, str) for c in columns]):
        task_ts = df[columns[0]].values
        labels = df[columns[1]].values
    elif all([isinstance(c, int) for c in columns]):
        task_ts = df.values[columns[0]]
        labels = df.values[columns[1]]
    else:
        err = "Invalid types for parameter columns elements, elements must all be str or all int."
        raise ValueError(err)
    
    if use_events is not None:
        event_locs = np.sort(np.hstack([np.where(labels==e)[0] for e in use_events]))
        task_ts = task_ts[event_locs]
        labels = labels[event_locs]
        
    map_ts, map_epoch_idx = get_nearest_timestamps(base_timestamps=eeg_ts,
                                                   target_timestamps=task_ts,
                                                   map_type=map_type)
    
    return np.stack([map_epoch_idx, labels], axis=1).astype(int)

# debug function
# def debug_original_samples(file_path, epoch_labels, epoch_ts):
#     # START DEBUG
#     sp = join(dirname(file_path), "state-info.csv")
#     state_info = pd.read_csv(sp, header=0, index_col=None)
#     try:
#         state = state_info['rewards'].values
#     except Exception:
#         state = state_info['labels'].values
#     try:
#         inter1 = np.where(state == 1)[0].flatten()
#         inter2 = np.where(state == 2)[0].flatten()
#     except Exception:
#         inter1 = group(np.where(state == 1)[0].flatten())
#         inter2 = group(np.where(state == 2)[0].flatten())
#         print(type(inter1))

#     test1 = np.where(epoch_labels==1)[0].flatten()
#     test2 = np.where(epoch_labels==2)[0].flatten()
    
#     print(state_info['timestamps'].iloc[inter1].values)
#     print(epoch_ts[test1])
    
#     print(state_info['timestamps'].iloc[inter2].values)
#     print(epoch_ts[test2])
#     set_trace()     
#     # END DEBUG