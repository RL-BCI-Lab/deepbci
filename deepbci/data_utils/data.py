from ast import literal_eval
from copy import deepcopy, copy
from functools import partial
from collections import defaultdict
from pdb import set_trace

import numpy as np
import pandas as pd
import mne

import deepbci
import deepbci.data_utils as dutils
import deepbci.utils.logger as logger

class DataGroup(object):
    """
        Data class for storing data, labels, and other meta information.
    """
    def __init__(self, data, labels, *, tags, event_ids=None, epochs=None):
        self.data = data
        self.labels = labels
        self.tags = tags
        self.event_ids = event_ids
        self.epochs = epochs
        self.metadata = defaultdict(list)
        
    def add_metadata(self, data_dict):
        for key, value in data_dict.items():
            self.metadata[key].append(value)
    
    def __repr__(self):
        return f"DataGroup({self.data.__class__.__name__}, {self.labels.__class__.__name__})"
    
    def __getitem__(self, index):
        return self.data[:, index], self.labels[index]

class Groups(object):
    """ Stores DataGroup objects using Pandas multi-index for easy indexing of data.
    
        Groups main focus is on utilizing pandas.Series multi-indexing feature for storing
        data based on their group, dataset, subject, and trial names. This allows building
        of new datasets where specific data can be accessed and mutated. 
        
        This class essentially acts as a wrapper for pandas.Series with multi-indexing. 
        
        Args:
            data_groups (dict): A dictionary where the key is the desired name of the 
            group and the value is a list of DataGroup objects. If the value is a list of
            lists then the nested listed will be combined into a single list when the
            data_map is built.
            
                Example input:
                    data_groups = {
                        'train': [DataGroup1, DataGroup2], 
                        'Valid': [[DataGroup1, DataGroup2], DataGroup3]
                    }
        
        Attributes:
            data_map (pd.Series): Pandas series with multi-indexing used to store all
                loaded data.
                
            _levels (list): A list of string names which specify what each multi-index 
                level will be called.
                
            _compressed_key (str): a string that represents which data levels
                have been compressed. Do not use np.NaN or None. This is because 
                pd.Series.drop() when indexed with an index containing None or NaN
                will not actually be dropped. However, it is possible to use 'NaN' 
                casted as a string.

    """
    
    def __init__(self, data_groups):
        self.data_map = None
        self._levels = ['group', 'dataset', 'subject', 'trial']
        self._compressed_key = 'NaN'
        self._build_data(data_groups)
    
    def __setitem__(self, index, value):
        self.data_map.loc[index] = value
        # Note: If performance degrades when adding to Series check impact of sort_index()
        self.data_map.sort_index()
        
    def __getitem__(self, index):
        """ Returns an array of deepbci.data_utils.DataGroup objects

            Args:
                index (list, tuple): A list or tuple of Pandas multi-index labels.
        
        """
        # Convert multi-index into proper format
        index = tuple([[str(i)] if not isinstance(i, (list, tuple)) else list(map(str, i)) for i in index])
        selected = self.data_map.loc[index]
        
        if hasattr(selected, 'ravel'):
            return selected.ravel()
        
        return selected

    def __repr__(self):
        """ Return a string representation for a particular DataFrame. """
        return self.data_map.__repr__()

    def _build_data(self, data_groups):
        df_data = []
        multi_idx = []
        
        for group, data in data_groups.items():
            data_groups[group] = list(np.hstack(data))

        for group, group_data in data_groups.items():
            for dg in group_data:
                df_data.append(dg)
                d, s, t = dg.tags['dataset'], dg.tags['subject'], dg.tags['trial']
                idx_names = (str(group), str(d), str(s), str(t))
                multi_idx.append(idx_names)
                
        indexes = pd.MultiIndex.from_tuples(multi_idx, names=self._levels)
        # WARNING: If performance degrades when building Series check impact of sort_index() 
        self.data_map = pd.Series(df_data, index=indexes, name='data').sort_index()

    def deepcopy(self, select=None):
        new_grps = deepcopy(self)
        
        selected = self.get_series(select)
        selected_dcp = deepcopy(selected.to_dict())
        
        new_grps.data_map = pd.Series(selected_dcp, name='data').sort_index()
        new_grps.data_map.index.names = self._levels
        
        return new_grps
    
    # def apply_class(self, class, select=None):
    #     if not isinstance(class, list):
    #         class = [class]
            
    #     selected_data = self.get_series(index=select)
        
    #     for c in class:
    #         (class_name, kwargs), = c.items()
    #          kwargs = kwargs if kwargs is not None else {}
             
    #         if isinstance(class_name, str):
    #             class_ref =  self._get_callable_func(class_name)
        
    #     class_inst = class_ref(**kwargs)
    #     class_inst.fit(selected_data)
    #     [class_inst.transform(d) for d in selected_data]
       
            
    def apply_func(self, func, select=None):
        """ Applies a callable to the data_map or a sub-set of the data_map.
        
            To change the value of information stored in each GroupData instance, the
            callable must update the values in place. 
            
            Args:
                func (callable): A callable object like a function that will edit the
                    data in place. This callable is required to take in the arguments
                    'data' and 'key' (data is the first positional, key is the second). 
                    The 'data' argument is the data for current DataGroup.
                    The 'key' argument is the multi-index for the current DataGroup.
                
                select (list): List of strings that are used to index the pd.Series 
                    multi-indexes to select specific data to be mutated. If none is passed
                    then all the data within self.data_map is used.
        """
        selected_data = self.get_series(index=select)

        if not callable(func):
            err = f"Object passed to func `{func}` is not callable. " \
                    f"Please pass a callable object."
            raise TypeError(err)

        [func(d[0], k) for k, d in selected_data.groupby(level=self._levels)]
    
    def apply_method(self, method, select=None):
        """ Apply methods for the current object type stored inside DataGroup.data.
            
            WARNING: Not all mne/np.ndarray methods are guaranteed to work and can cause 
            issues later on down the line if you are using DBCI's DataGroup and Groups objects. 
            For instance, if you use mne.Epoch.drop_bads() method instead of DBCI's mutators 
            wrapper function reject(), then labels will not be updated to match dropped 
            data. This method works great if you are only calling methods that modify 
            DataGroup.data values. Otherwise it may be advisable to manually call methods
            on data contained within Groups to account for adverse effects!
            
            Args:
                method (dict): Dictionary the contains the method name which is stored in
                    the 'name' key and any additional method kwargs stored in the 'kwargs'
                    key (optional).
                
                select (list): List of strings that are used to index the pd.Series 
                    multi-indexes to select specific data to be mutated. If none is passed
                    then all the data within self.data_map is used.
        """           
        selected_data = self.get_array(index=select)
        
        method_name = method['name']
        kwargs = method['kwargs'] if method.get('kwargs') is not None else {}
        
        for d in selected_data:
            bound_method = getattr(d.data, method_name)
            bound_method(**kwargs)
                   
    def compress(self, compress_level, inplace=True, select=None, **kwargs):
        """ Compresses data map based on the four multi-index levels. 
                             
            WARNING: If you have DataGroup.epochs set, compressing will cause you to lose 
            the data stored in the DataGroup.epochs as it can not be compressed easily
            Likewise, DataGroup.tags will be compressed in a lossly manner.Therefore, tag 
            information can be lost if all datasets do not have the same subjects and trials.
            
            Args:
            
                compress_level (tuple): Multi-index level name that will be compressed.
                    All levels below the specified level will also be compressed into
                    a single deepbci.data_utils.data.DataGroup object. See self.levels for
                    the names of each multi-index levels.
                
                inplace (bool): Perform in-place mutation of Groups if true, else return
                    the new data_map.
                    
                select (list): A list of of multi-indexes (for single 
                    index only a str is need, for multiple then pass lists/tuples) for 
                    indexing data_map a pd.Series. See pd.Series multi-index indexing for 
                    more information. 
        """
        compress_level_idx = self._levels.index(compress_level)+1
        groupby_levels = self._levels[:compress_level_idx]
        
        # Make sure select is a tuple consisting of lists so when indexing the data_map
        # indexed levels are kept.
        selected = self.get_series(select)
        comp_groups = selected.groupby(level=groupby_levels)

        multi_idx = []
        idx_data = []
        for name, data in comp_groups:
            if not isinstance(name, tuple):
                name = (name,)
                
            add_nans = (self._compressed_key,) * (len(self._levels) - len(name[:compress_level_idx]))
            comp_name = name[:compress_level_idx] + add_nans
   
            comp_data = compress_data(data.values.ravel(), **kwargs)
            
            multi_idx.append(comp_name)
            idx_data.append(comp_data)

        multi_index = pd.MultiIndex.from_tuples(multi_idx, names=self._levels)
        data_map = pd.Series(idx_data, index=multi_index, name='data').sort_index()
        
        if inplace:
            if select is not None:
                self.data_map = pd.concat([self.data_map, data_map])
                # self.data_map = self.data_map.append(data_map)
                del_idx = selected.index
                self.data_map.drop(del_idx, inplace=True)
                self.data_map.sort_index(inplace=True)
            else:
                self.data_map = data_map
            return self
       
        new_grp = copy(self)
        new_grp.data_map = data_map

        return new_grp
    
    def get_series(self, index=None):
        """ Selects all data or specific data from the data_map attribute and returns
            as a pd.Series of data_utils.data.DataGroup objects.
            
            Args:
                index (list): A list of of multi-indexes (for single 
                    index only a str is need, for multiple then pass lists/tuples) for 
                    indexing data_map a pd.Series. See pd.Series multi-index indexing for 
                    more information. 
            Return:
                DataGroup selected via multi-indexing formatted as pd.Series
        """
        if index is not None:
            index = tuple([[str(i)] if not isinstance(i, list) else map(str, i) for i in index])
            return self.data_map.loc[index] 
        else:
            return self.data_map
    
    def get_array(self, index=None): 
        """ Selects all data or specific data from the data_map attribute and returns
            as a array of data_utils.data.DataGroup objects

            Args:
                index (list): A list of lists or a list of strings that are formatted to
                    index a multi-index pandas.Series. For more information
                    on the format see the following pandas multi-index documentation:
                    https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html.
                    All indexes utilize the pandas.IndexSlice wrapper, i.e. format according
                    to the pandas.IndexSlice format.
            
            Return
                DataGroup selected via multi-indexing formatted as an np.ndarray
                    
        """
        if index is None:
            return self.data_map.values
        else:
            selected = self[index]
            if len(selected) == 0:
                err = "The following index {} was not found in the data_map"
                raise IndexError(err.format(index))
            return selected
        
    def get_level_values(self, level, select=None):
        if select is not None:
            selected = self.get_series(select)
        else:
            selected = self.data_map
             
        return selected.index.get_level_values(level=level)
    
    @property
    def levels(self):
        return self._levels
    
    @property
    def compressed_key(self):
        return self._compressed_key
        
def run_group_mutators(groups, mutate):
    """ Run Groups methods automatically. 
        
        Useful for building data by running apply_func or apply_method methods for
        all specified data within the Groups object.  
        
        Args:
            groups (Groups): A Groups object.
            
            mutate (list): A list of dictionaries where the key is the a method to run
                for the Groups object and the values are dictionaries containing 
                parameters.
    """
    for func_or_meth in mutate:
        apply_ = [*func_or_meth]
        if len(apply_) > 1:
            err = f"More than one apply func/method was given {apply_}"
            raise RuntimeError(err)
        else:
            apply_ = apply_[0]
        
        kwargs = func_or_meth[apply_] if func_or_meth[apply_] is not None else {}
        try:
            bound_apply = getattr(groups, apply_)
            bound_apply(**kwargs)
        except (KeyError, IndexError) as e:
            warn = f"Function or method {bound_apply.__name__} was unable to index data: {kwargs}. This is " \
                   f"most likely due to select index not being found! Continuing..."
            logger.warn(warn)
            logger.warn(e)
        
def compress_data(data, **kwargs):
    """ Compresses data based on DataGroup.data type.
    
        The current supported types for compression are mne.io.Raw, mne.Epoch,np.ndarray
        or variants of the aforementioned types.    
        
        Args:
            data (list): List of deepbci.data_util.data.DataGroup objects.
        Returns:
            A new DataGroup object which contains all of the original list of DataGroup objects.
    """
    assumed_type = data[0].data

    if isinstance(assumed_type, (mne.io.Raw, mne.io.RawArray)):
        compress_func = compress_data_raw
    elif isinstance(assumed_type, (mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs)):
        compress_func = compress_data_epoch
    elif isinstance(assumed_type, np.ndarray):
        compress_func = compress_data_array
    else:
        err = """Invalid type {} for compressing. Make sure all DataGroup.data types in \
            data_map are all of one of the following types: mne.io.Raw, mne.Epochs, \
            np.ndarray.""".format(assumed_type)
        raise TypeError(err)

    return compress_func(data, **kwargs)
    
def compress_data_raw(data, **kwargs):
    """ Compresses Groups.data when DataGroup.data is of type mne.io.Raw.
    
        Note:
            Epoch concatentation is based on mne.concatenate_events
    """
    extracted_data = []
    extracted_epochs = []
    comp_ids = {}
    comp_tags = dict(dataset=set(), subject=set(), trial=set())
    
    offset = 0 + data[0].data.first_samp
    for d in data:
        if not isinstance(d.data, (mne.io.Raw, mne.io.RawArray)):
            err = "Not all DataGroup.data objects are of type mne.io.Raw"
            raise TypeError(err)
        
        if d.data.get_data().shape[-1] != 0:
            extracted_data.append(d.data)
            compress_tags(d.tags, comp_tags)
            compress_event_ids(d.event_ids, comp_ids)

            if d.epochs is not None:
                epochs = np.hstack([d.epochs[:, 0, None] + offset, d.epochs[:, 1:]])
                extracted_epochs.append(epochs)
                # print(d.tags, offset, np.concatenate(extracted_epochs, axis=0).shape)
                offset += d.data.last_samp - d.data.first_samp +1
        else: 
            logger.warn("Skipped: {} due to shape {}".format(d.tags, d.data.get_data().shape))
 
    comp_data = mne.concatenate_raws(extracted_data, **kwargs)
    comp_labels = comp_data['stim']
    comp_epochs = np.concatenate(extracted_epochs, axis=0) if d.epochs is not None else None
    
    new_data = DataGroup(data=comp_data, 
                    labels=comp_labels, 
                    tags=comp_tags, 
                    event_ids=comp_ids,
                    epochs=comp_epochs)

    return new_data

def compress_data_epoch(data, **kwargs):
    """ Compresses Groups.data when DataGroup.data is of type mne.Epoch."""
    extracted_data = []
    extracted_ids = []
    comp_ids = {}
    comp_tags = dict(dataset=set(), subject=set(), trial=set())
    
    for d in data:
        if not isinstance(d.data, (mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs)):
            err = "Not all DataGroup.data objects are of type mne.Epochs"
            raise TypeError(err)
        
        if len(d.data) != 0:
            extracted_data.append(d.data)
            extracted_ids.append(d.event_ids)
            compress_tags(d.tags, comp_tags)
            compress_event_ids(d.event_ids, comp_ids)
        else: 
           logger.warn("Skipped {} for compression due to a length of {}".format(d.tags, len(d.data)))

    comp_data =  mne.concatenate_epochs(extracted_data, **kwargs)
    comp_labels = comp_data.events[:, -1]
    comp_epochs = comp_data.events
    
    new_data = DataGroup(data=comp_data, 
                    labels=comp_labels, 
                    tags=comp_tags, 
                    event_ids=comp_ids,
                    epochs=comp_epochs)
    
    return new_data

def compress_data_array(data, axis=0):
    """ Compresses Groups.data when DataGroup.data is of type np.ndarray.
    
        WARNING: DataGroup.epochs information will be lost upon compression!
    """
    extracted_data =  []
    extracted_labels = []
    # extracted_epochs = []
    comp_ids = {}
    comp_tags = dict(dataset=set(), subject=set(), trial=set())
    concat_method = partial(np.concatenate, axis=axis)
    
    for d in data:
        if not isinstance(d.data, np.ndarray):
            err = "Not all DataGroup.data objects are of type np.ndarray"
            raise TypeError(err)
        
        if len(d.data) != 0:
            extracted_data.append(d.data)
            extracted_labels.append(d.labels)
            compress_tags(d.tags, comp_tags)
            compress_event_ids(d.event_ids, comp_ids)
        else: 
           logger.warn("Skipped {} for compression due to a length of {}".format(d.tags, len(d.data)))
                 
    comp_data =  np.concatenate(extracted_data, axis=axis)
    comp_labels = np.concatenate(extracted_labels)
    
    new_data = DataGroup(data=comp_data, 
                    labels=comp_labels, 
                    tags=comp_tags, 
                    event_ids=comp_ids)

    return new_data
    
def compress_tags(tags, new_tags):
    """ Method for compressesing DataGroup.tags into a single tag. """
    for tag_k, tag_v in tags.items():
        if not isinstance(tag_v, (list, tuple, set)):
            tag_v = {tag_v}
        elif isinstance(tag_v, (list, tuple)):
            tag_v = set(tag_v)
          
        new_tags[tag_k].update(tag_v)
        
def compress_event_ids(event_ids, new_event_ids):
    """ Method for compressesing DataGroup.event_ids into a single tag.
        
        WARNING: If there are multiple DataGroup.event_ids with the same key and different
        values then the last value will be used.
    """
    for id_k, id_v in event_ids.items():
        if id_k in new_event_ids:
            continue
        elif id_v in [*new_event_ids.values()]:
            err = "Error compressing event ids. There is already a key with the given value {}."
            raise RuntimeError(err.format(id_v))
        else:
            new_event_ids[id_k] = id_v
            