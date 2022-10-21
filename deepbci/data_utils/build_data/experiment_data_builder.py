import sys
import os
from pdb import set_trace
from os.path import join
from dataclasses import dataclass, asdict, field

import pandas as pd
import numpy as np

import deepbci
import deepbci.utils.utils as utils
from deepbci.utils.utils import get_module_root
from deepbci.data_utils.build_data.experiment_data import ExperimentData

class ExperimentDataBuilder():
    """ Builds raw DeepBCI datasets by combining EEG recording and task related information 
        into a single dataset.  
    """
    def __init__(
        self,
        data_dir_template=["data", "{}", "S{}", "trial-{}"],
        eeg_template='OpenBCI-RAW',
        state_info_template='state-info',
        task_info_template = 'game-info',
        time_template='time-{}',
        save_template=['filtered', '{}'],
        verbose=False,
    ):
        self._module_root= utils.get_module_root()
        self._data_dir_tmplt = join(*data_dir_template)
        self._eeg_tmplt = eeg_template
        self._state_info_tmplt= state_info_template
        self._task_info_tmplt = task_info_template
        self._time_tmplt = time_template
        self._save_tmplt = join(*save_template)
        self.verbose = verbose
    
    # TODO: Remove hardcoded filter and allow config to specify filter function
    # TODO: Directly add build_label_kwargs which are nearest_kwargs here.
    def build_to_disk(self, datasets, build_kwargs, file_name=None, dry_run=False):
        for task_name, task_cfg in datasets.items():
            for sub_name, sub_cfg in task_cfg['subjects'].items():
                for trial_name in sub_cfg['trials']:
                    file_paths, data_dir = self._build_paths(task=task_name,
                                                             subject=sub_name, 
                                                             trial=trial_name,
                                                             signal_types=task_cfg['signal_types'])

                    if self.verbose: print(f"\nEEG Path: {file_paths['eeg_path']}")
                    raw_data = ExperimentData(**task_cfg['exp_data_kwargs'], **file_paths)
                    self._build(raw_data=raw_data, **build_kwargs)
                    if self.verbose: self.info_dump(raw_data)
                    
                    # Build data save path
                    filter_info = build_kwargs.get('butter_filter_kwargs', {})
                    
                    rel_save_path = self._build_rel_save_path(
                        file_name=file_name, 
                        filter_info=filter_info
                    )
                    abs_save_path = join(data_dir, rel_save_path)
                    
                    # Save file
                    if dry_run == False:
                        self._write_to_file(raw_data=raw_data, 
                                            electrodes=task_cfg['electrodes'], 
                                            save_path=abs_save_path)
                        print("Save path: {}".format(abs_save_path))
                    else:
                        print("Dry run save path: {}".format(abs_save_path))

    def _build(self,
               raw_data,
               build_labels_kwargs,
               butter_filter_kwargs=None):
        
        raw_data.load()
        if butter_filter_kwargs is not None:
            raw_data.butter_filter(**butter_filter_kwargs)
        raw_data.sync_data()
        raw_data.build_labels(**build_labels_kwargs)
    
    def _build_paths(self, task, subject, trial, signal_types):
        file_paths = {}
        data_dir = join(
            os.path.sep, 
            self._module_root, 
            self._data_dir_tmplt
        ).format(task, subject, trial)

        # Find OpenBCI-RAW file
        file_paths['eeg_path'] = utils.path_to(data_dir, self._eeg_tmplt)
        
        # Find state-info.csv file
        file_paths['state_info_path'] = utils.path_to(data_dir, self._state_info_tmplt)
        
        # Find time-<signal>.csv files
        event_paths = {}
        for signal, label in signal_types.items():
            event_path = self._time_tmplt.format(signal)
            event_paths[label] = utils.path_to(data_dir, event_path)
        file_paths['event_paths'] = event_paths
        
        # Fine game-info.csv file
        file_paths['task_info_path'] = utils.path_to(data_dir, self._task_info_tmplt)

        return file_paths, data_dir
        
    def _build_rel_save_path(self, file_name=None, filter_info=None):
        """ Build the relative save path for once built EEG data."""

        if file_name is None:
            file_name = self._build_file_name(**filter_info)
        else:
            file_name = f'{file_name}.csv'
            
        return self._save_tmplt.format(file_name)
    
    def _build_file_name(self, ftype=None, order=None, lowcut=None, highcut=None):
        if ftype is None:
            file_name = "eeg.csv"
        elif ftype.lower()== 'butter_bandpass_filter':
            file_name = "eeg-order{}-low{}-high{}.csv".format(order, lowcut, highcut)
        elif ftype.lower() == 'butter_highpass_filter':
            file_name = "eeg-order{}-low{}.csv".format(order, lowcut)
        elif ftype.lower() == 'butter_lowpass_filter':
            file_name = "eeg-order{}-high{}.csv".format(order, highcut)
        else:
            raise ValueError("Invalid butter filter given: {}".format(self.gen['type']))
        return file_name
    
    def _write_to_file(self, raw_data, electrodes, save_path):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data = [
            raw_data.eeg_labels, 
            raw_data.eeg, 
            raw_data.eeg_ts[:, None]
        ]
        data = np.concatenate(data, axis=1)
        column_names = ['labels', *electrodes, 'timestamps']
        df = pd.DataFrame(data, columns=column_names)
        df.labels = df.labels.astype(int)
        df.to_csv(save_path, header=True, index=False)
             
    def info_dump(self, exp_data: ExperimentData):
        if exp_data.fill_info is not None and len(exp_data.fill_info) > 0:
            info_str = ""
            for _, info in exp_data.fill_info.items():
                info_str = f"Missing: {len(info['data'])}, Start: {info['start']}, " \
                           f"Start offset: {info['start_offset']}, End offset: {info['end_offset']}, " \
                           f"Start iter: {info['siter']}, End iter: {info['eiter']}"
                data = np.hstack([np.vstack(info['data']), info['utils']])
                info_str += f"\n{info_str}\n{data}"
            print(utils.bordered(f"Filling missing info:" + info_str))
             
        exp_data.eeg_info.print_info()
        exp_data.task_info.print_info()
        
        print("Additional Info:")
        print(f"\teeg shape: {exp_data.eeg.shape}")
        print(f"\teeg ts shape: {exp_data._eeg_ts.shape}")
        print(f"\ttask states shape: {exp_data._task_state.shape}")
        print(f"\ttask ts shape: {exp_data._task_ts.shape}")
        print(f"\tstart padding: {exp_data.before_padding}")
        print(f"\tbuilt padded start: {exp_data._eeg_ts[0]}")
        print(f"\tend padding: {exp_data.after_padding}")
        print(f"\tbuilt padded end: {exp_data._eeg_ts[-1]}")