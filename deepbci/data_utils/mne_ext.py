from pdb import set_trace

import numpy as np
import mne

def add_stim_channel(raw, fs):
    info = mne.create_info(
        ch_names=['STI'],
        sfreq=fs,
        ch_types=['stim']
    )
          
    shape = (1, len(raw))
    stim_placeholder = np.zeros(shape)
    stim_raw = mne.io.RawArray(stim_placeholder, info, verbose='ERROR')
    raw.add_channels([stim_raw], force_update_info=True)
    
def generate_events(event_indices, event_labels):
    prelabels = np.zeros(event_indices.shape)
    return np.stack([event_indices, prelabels, event_labels], axis=1).astype(int)

def get_raw_stim(raw, use_events=None):
    events, _ = mne['stim']
    return _extract_events(events, use_events)


def get_epoch_events(epoch, use_events=None):
    events = epoch.events[:, -1]
    return _extract_events(events, use_events)

def _extract_events(events, use_events):
    event_dict = {}
    for e in use_events:
        e_idx = np.where(events==e)[0]
        event_dict[e]= e_idx
    
    return event_dict
