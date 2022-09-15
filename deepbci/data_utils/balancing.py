from pdb import set_trace
from collections.abc import Iterable  

import numpy as np

from deepbci.utils import logger

def downsample(labels, rate=1):
    """ Downsample majority classes to the a given ratio of the minority class.
        Assumes that the last dimension of labels represents the positive class
        for one-hot encoded labels.
    
        Notes:
            If a rate of 1 is passed the class with the most samples is
            downsampled to the class with the largest class.

            Rate == 1 implies a balanced dataset where the majority class is
            downsampled to the size of the minority.

            Rate > 1 implies downsampling the majority class where the majority
            still maintains the majority samples.

            Rate < 1 implies downsampling the majority class where the minority
            becomes the minority 

            Rate <= 0 means that the majority class isn't downsampled at all.
        
        Args:
            labels (np.ndarray): NumPy array of binary labels of shape (N, 2), (N, 1)
            or (N,). Where N are the number of samples.

        Returns:
            Indexes of the data corresponding to the ratio given.
    """
    if rate <= 0: return
    
    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    elif len(labels.shape) > 2:
        err = "Passed labels has shape {}. downsample() only works with binary labels"
        raise ValueError(err.format(labels.shape))
    if len(np.unique(labels)) != 2:
        err = "Labels has more or less than 2 unique labels. downsample() only works with " \
              "binary labels"
        raise ValueError(err)
    
    label_ids = np.unique(labels)
    
    min_class, maj_class, class_idx = _get_binary_classes(labels, label_ids)

    # Max dataset size for maj
    sample_size = int(len(class_idx[min_class]) * rate)
    # Randomly select indexes
    maj_idxs = np.random.choice(class_idx[maj_class], size=sample_size, replace=False)

    return np.hstack([maj_idxs, class_idx[min_class]])

def upsample(labels, rate=1, verbose=False):
    """ Upsample the minority class to the a given ratio of the majority class.

        This function works on binary labels ONLY!
    
        Notes:
            If a rate of 1 is passed the class with the most samples is
            downsampled to the class with the largest class.

            Rate == 1 implies a balanced dataset where the minority class is
            upsampled to the size of the minority.

            Rate > 1 implies upsampling the minority class where the minority
            becomes the majority.

            Rate < 1 implies upsampling the minority class where the majority
            maintains its majority status.

            Rate <= 0 means that the minority class isn't upsampled at all.
            
        Args:
            labels (np.ndarray): NumPy array of binary labels of shape (N, 2), (N, 1)
            or (N,). Where N are the number of samples.
        
        Returns:
            Indexes of the data corresponding to the ratio given.
    """
    if rate <= 0: return

    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    elif len(labels.shape) > 2:
        err = "Passed labels has shape {}. upsample() only works with binary labels"
        raise ValueError(err.format(labels.shape))
    if len(np.unique(labels)) != 2:
        err = "Labels has more or less than 2 unique labels. upsample() only works with " \
              "binary labels"
        raise ValueError(err)
    
    label_ids = np.unique(labels)

    min_class, maj_class, class_idx = _get_binary_classes(labels, label_ids)

    # Target size for min data
    sample_size = int(len(class_idx[maj_class]) * rate)
    # Determine the number repeats for the min data.
    reps = sample_size // len(class_idx[min_class])
    # Update sample size with leftovers from repeatting min data
    remainder = sample_size - reps * len(class_idx[min_class])

    # Construct upsampled data indexes for the min class
    min_idxs = np.repeat(class_idx[min_class], reps, axis=0)
    if remainder != 0:
        filler = np.random.choice(class_idx[min_class], size=remainder, replace=False)
        min_idxs = np.hstack((min_idxs, filler))
    
    sampled_idx = np.hstack([class_idx[maj_class], min_idxs])
    
    if verbose:
        logger.info("Maj shape: {}".format(class_idx[maj_class].shape))
        logger.info("Min shape: {}".format(min_idxs.shape))
        logger.info("Total shape: {}".format(sampled_idx.shape))
        
    return sampled_idx

def _get_binary_classes(labels, label_ids):
    positive = _find_labels(labels, label_ids[0])
    negative = _find_labels(labels, label_ids[1])
    
    class_idx = [positive, negative]
    min_class = np.argmin([len(positive), len(negative)], axis=0)
    maj_class = np.argmax([len(positive), len(negative)], axis=0)
    
    return min_class, maj_class, class_idx
    
def _find_labels(labels, targets):
    if not isinstance(targets, Iterable):
        targets = [targets]
        
    found = []
    for t in targets:
        found.append(np.where(labels == t)[0])
        
    return np.hstack(found)