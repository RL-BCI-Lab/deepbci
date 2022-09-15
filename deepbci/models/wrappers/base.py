import os
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
from sklearn import utils

class BaseDataset():
    _ele_type = np.ndarray
    
    def __init__(
        self,  
        trn_set=None, 
        vld_set=None, 
        tst_set=None,
        *,
        shuffle=False,
        seed=None,
    ):
        self._trn_set = trn_set
        self._vld_set = vld_set
        self._tst_set = tst_set
        self.shuffle = shuffle
        self.seed = seed

    def _check_set_type(self, x):
        if not isinstance(x, (type(None), tuple, list)):
            err = f"Input was found to be a {type(x)}, expected None, tuple, or list"
            raise TypeError(err)
        if isinstance(x, (tuple,list)):
            if len(x) != 2:
                raise ValueError("Dataset can must have 2 elements: (data, labels).")
            for ele in x:
                self._check_element_type(ele)
        return x
    
    def _check_element_type(self, ele):
        if not isinstance(ele, self._ele_type):
            err = f"Element in set was found to be a {type(ele)}: expected {self._ele_type}"
            raise TypeError(err)
        
    @property
    def trn_set(self):
        return trn_set
        
    @property
    def vld_set(self):
        return vld_set

    @property
    def tst_set(self):
        return tst_set
        
    @trn_set.getter  
    def trn_set(self):
        if self._trn_set:
            return self.build_dataset(*self._trn_set, shuffle=self.shuffle)
        else:
            return None
        
    @vld_set.getter 
    def vld_set(self):
        if self._vld_set:
            return self.build_dataset(*self._vld_set, shuffle=False)
        else:
            return None

    @tst_set.getter 
    def tst_set(self):
        if self._tst_set:
            return self.build_dataset(*self._tst_set, shuffle=False)
        else:
            return None
    
    @trn_set.setter  
    def trn_set(self, value):
        self._trn_set = self._check_set_type(value)
    
    @vld_set.setter 
    def vld_set(self, value):
        self._vld_set = self._check_set_type(value)
        
    @tst_set.setter 
    def tst_set(self, value):
        self._tst_set = self._check_set_type(value)
        
    def build_dataset(self, data, labels, shuffle):
        """ Build and prepare dataset using NumPy arrays.
        
            Args:
                data (np.ndarray): Data given as a NumPy array.
                
                labels (np.ndarray): Labels given as a NumPy array.
                
                shuffle (bool): Determines if data should be shuffled.
        """
        if shuffle:
            data, labels = utils.shuffle(data, labels, random_state=self.seed)
        return data, labels
        
class BaseWrapper(ABC):
    """ Base abstract class wrapper for wrapping a given model

        All model wrappers should implement this class so logging functionality works.
    """
    _dataset = BaseDataset
    
    @abstractmethod
    def fit(self, trn_set, vld_set=None, class_weights=None):
        return
    
    @abstractmethod
    def predict(self, tst_set):
        return preds
    
    @abstractmethod
    def save(self, filepath):
        return
    
    @abstractmethod
    def load(self, filepath):
        return
    
    @staticmethod
    def preinstantiate(**kwargs):
        return
    
    @property
    def dataset(self):
        return self._dataset 

    def _make_dir(self, filepath):
        head, _ = os.path.split(filepath)
        if head and not os.path.exists(head):
            os.makedirs(head)