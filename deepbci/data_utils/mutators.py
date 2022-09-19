from pdb import set_trace

import numpy as np
import mne
import scipy.linalg

import deepbci.data_utils.balancing as balancing
from deepbci.utils.class_weight import compute_class_weight
from deepbci.utils import utils, logger

def balance(data, key, btype, **kwargs):
    """ Wrapper function for data_utils.balancing functions for balancing minority and majority 
        classes.
    
        This function only works on Data objects who's Data.data variable is of type
        np.ndarray. Like all other functions and classes within the mutator namespace 
        this function modifies the Data object in-place.
        
        Args:
            data (Data): Data object where Data.data is of type np.ndarray.
            
            btype (str): Name of balancing function to use from data_utils.balancing. For
                instance 'upsample' corresponds to the upsample() function.
    
    """
    if not isinstance(data.data, np.ndarray):
        raise ValueError("Data.data needs to of type np.ndarray in order to upsample")
    
    if len(data.labels) != len(data.data):
        raise ValueError("Can not upsample, the length of labels and data samples mismatch.")
    
    btype = getattr(balancing, btype)
    sampled_idx = btype(data.labels, **kwargs)

    data.data = data.data[sampled_idx]
    data.labels = data.labels[sampled_idx]
    
    if data.epochs is not None:
        data.epochs = data.epochs[sampled_idx]

def epoch(data, key, **kwargs):
    """ Epoch wrapper for changing mne.Raw to an mne.Epochs instance.
    
        This function only works on Data objects who's Data.data variable is of type
        mne.Raw (or another variant of Raw). Like all other functions and classes within 
        the mutator namespace this function modifies the Data object in-place. 

        WARNING: mne.Epochs will drop data at the end of the timeseries
        as there wont been enough data remaining to fill the window. Use
        epochs.drop_log to see dropped epochs or "bad" epochs.
        
        Args:
            data (Data): Data object where Data.data is of type mne.io.Raw. Data.epochs
                also needs to be initialized before calling this function.
            
            epoch_kws (dict): Parameters for mne.Epochs initialization.
    """
    if not isinstance(data.data, (mne.io.Raw, mne.io.RawArray)):
        raise ValueError("Data.data needs to of type mne.io.Raw in order to epoch")
    
    if data.epochs is None:
        raise ValueError("Data.epochs must be initialized before epoching")
    epochs = mne.Epochs(data.data, data.epochs, **kwargs)
    
    data.data = epochs
    data.labels = epochs.events[:, -1] # set new labels 
    data.epochs = epochs.events

def remap_labels(data, key, label_map):
    """ Remaps labels to a new label.
    
        Like all other functions and classes within the mutator namespace this function
        modifies the Data object in-place.
        
        Args:
            data (Data): A deepbci.data_utils.data.Data object where Data.data is of type 
                np.ndarray.
            
            label_map (dict): Dictionary where the key represents the original label and
                the value represents the desired new label.
    """
    if not isinstance(label_map, dict):
        raise ValueError("label_map is not of type dict")

    for k, v in label_map.items():
        np.place(data.labels, np.in1d(data.labels, k), v)

def transpose(data, key, new_shape):
    """ Transpose Data.data dimensions
    
        The default dimension order follows suit with mne where: mne.Raw has shape 
        (chs, samples) and mne.Epoch has (epochs, chs, window). 
    """
    if not isinstance(data.data, np.ndarray):
        raise ValueError("Data.data needs to of type np.ndarray in order to transpose")
    
    data.data =  data.data.transpose(new_shape)
                                  
def reject_epochs(data, key, **kwargs):
    """ Artifact rejection wrapper for mne.Epoch method drop_bad.
    
        Like all other functions and classes within the mutator namespace this function
        modifies the Data object in-place.
        
        Args:
            data (dbci.data_utils.Data): Data object where Data.data is of type mne.Epochs.
                
            kwargs (dict): Parameters for mne.Epochs.drop_bad. parameters can be
                overridden at a subject level if data is of type list and subject_overrides
                if passed.

    """
    if not isinstance(data.data, (mne.Epochs, mne.EpochsArray, mne.epochs.BaseEpochs)):
        raise ValueError("Data.data must be of type mne.Epochs")
    # logger.info(f"Rejecting epochs for key: {key}")
    dropped = data.data.drop_bad(**kwargs)
    data.labels = data.data.events[:, -1]
    data.epochs = data.data.events
    
def to_numpy(data, key, **kwargs):
    """ Converts mne.Raw and mne.Epochs to NumPy arrays.
    
        Like all other functions and classes within the mutator namespace this function
        modifies the Data object in-place.
        
    """
    valid_types =  (
        mne.io.Raw, 
        mne.io.RawArray,
        mne.Epochs, 
        mne.EpochsArray,
        mne.epochs.BaseEpochs
    )
    if not isinstance(data.data, valid_types):
        raise ValueError("Data.data must be of type mne.io.Raw or mne.Epochs")
    
    data.data = data.data.get_data(**kwargs)
    
def expand_dims(data, key, axis):
    """ Wrapper for NumPy's np.expand_dim() function.
    
        Like all other functions and classes within the mutator namespace this function
        modifies the Data object in-place.
        
        Args:
            data (dbci.data_utils.Data): Data object where Data.data is of type np.ndarray.
        
    """
    if not isinstance(data.data, np.ndarray):
        raise ValueError("Data.data needs to of type np.ndarray in order to expand dimensions")
    
    data.data = np.expand_dims(data.data, axis=axis)
    
def compress_dims(data, key, start,  *, n=2):
    if not isinstance(data.data, np.ndarray):
        raise ValueError("Data.data needs to of type np.ndarray in order to expand dimensions")

    if n < 2:
        raise ValueError("The 'n' argument must be greater than 2: received n value of {n}")
    
    shape = data.data.shape
    data.data = np.reshape(data.data, shape[:start] + (-1,) + shape[start+n:])

def to_onehot(data, key):
    """ Convert labels to one-hot encodings
        
        WARNING: If you have class labels that are not increasing by 1 and starting from
        0 then the training/results may fail or be distorted if you use one-hot 
        conversion. For example, if we have the labels [0, 2] then when converting to a
        one-hot you will get [[1, 0], [0, 1]]. This means we can no longer recover the
        original labels.
        
        Args:
            data (dbci.data_utils.Data): DBCI Data object.
    """
    n_classes = len(np.unique(data.labels))

    data.labels = np.eye(n_classes)[data.labels]
    
def rescale(data, key, *, scaler=None, save=False, load=False, filepath=None):
    """ Rescales data features using Sklearn's Transformer class
    
        Args:
            data (dbci.data_utils.Data): DBCI Data object.
            
            scaler (Sklearn Transformer): A Sklearn Transformer or a custom class that uses
                Sklearn's ducktyping class structure.
                
            save (bool): If true, scaler class will be pickled and saved
            
            load (bool): If true, scaler class will be loaded from a pickle file
            
            filepath (str): Path to file used for saving or loading scaler
    
    """

    if load:
        if filepath is None:
            raise ValueError("Filepath must be set when loading a scaler class.")
        scaler = utils.pickle_load(filepath)
        data.data = scaler.transform(data.data)
    elif scaler:
        scaler.fit(data.data)
        data.data = scaler.transform(data.data)
        if save:
            if filepath is None:
                raise ValueError("Filepath must be set when saving a scaler class.")
            utils.pickle_dump(scaler, filepath)
    else:
        raise ValueError("A scaler was not provided and load was set to False. Please provide one or the other.")
    
def filter(data, key, **kwargs):
    """ Filter MNE data object given low pass and/or high pass bound
        
        MNE docs:
            Link for mne.io.Raw
                https://mne.tools/stable/generated/mne.io.RawArray.html#mne.io.RawArray.filter
            Link for np.ndarray:
                https://mne.tools/stable/generated/mne.filter.filter_data.html?highlight=mne%20filter%20filter_data#mne.filter.filter_data
            
            l_freq and h_freq are the frequencies below which and above which, 
            respectively, to filter out of the data. Thus the uses are:

            l_freq < h_freq: band-pass filter
            l_freq > h_freq: band-stop filter
            l_freq is not None and h_freq is None: high-pass filter
            l_freq is None and h_freq is not None: low-pass filter
    """
    valid_types =  (
        mne.io.Raw, 
        mne.io.RawArray,
        mne.Epochs, 
        mne.EpochsArray,
        mne.epochs.BaseEpochs
    )
    if isinstance(data.data, valid_types):
        data.data = data.data.filter(**kwargs)  
    elif isinstance(data.data, np.ndarray):
        data.data = mne.filter.filter_data(data.data, **kwargs)
    else:
        raise ValueError("Data.data needs to of type np.ndarray, mne.io.Raw, or mne.Epochs")
    
    
def log_class_info(data, key, *, class_weight='balanced', argmax_axis=1):
    if not isinstance(data.labels, np.ndarray):
        raise ValueError("Data.labels needs to of type np.ndarray in order to log class info")

    y = data.labels
    
    if len(y.shape) > 1:
        y = np.argmax(y, axis=argmax_axis)
    
    classes, counts = np.unique(y, return_counts=True)
    class_dict = {cls: c for cls, c in zip(classes, counts)}
    cw = compute_class_weight(y, class_weight, argmax_axis)
    
    logger.info(f"{data.tags}")
    logger.info(f"\tData count: {len(y)}")
    logger.info(f"\tClass balance: {class_dict}")
    logger.info(f"\tClass weights: {cw}")