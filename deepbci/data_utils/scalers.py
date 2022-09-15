from pdb import set_trace

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class STD(BaseEstimator, TransformerMixin):
    def __init__(self, axis=0):
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis
    
    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=self.axis, keepdims=True)
        self.std = np.maximum(np.std(X, axis=self.axis, keepdims=True), 1e-8)
    
    def transform(self, X):
        return (X - self.mean) / self.std

class MAD(BaseEstimator, TransformerMixin):
    def __init__(self, axis=0, **kwargs):
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis
        self.sci_kwargs = kwargs
    
    def fit(self, X, y=None):
        self.mad = stats.median_abs_deviation(X, axis=self.axis, **self.sci_kwargs)
        if 0 in self.mad:
            err = "MAD scaler detected a zero in the self.mad values. A divide by zero error is being thrown " \
                  "to prevent inf values. See this post on what MAD=0 means "\
                  "http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/"
            raise ZeroDivisionError(err)
        
    def transform(self, X):
        return X / self.mad

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, axis=0):
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis
    
    def fit(self, X, y=None):
        self.mins =  np.amin(X, axis=self.axis, keepdims=True)
        self.maxes = np.max(X, axis=self.axis, keepdims=True)
        self.diff = self.maxes - self.mins
        self.scale = np.maximum(self.diff, 1e-8)
        
    def transform(self, X):
        return (X - self.mins) / self.scale
    
class L2(BaseEstimator, TransformerMixin):
    def __init__(self,  axis=0):
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis
        
    def fit(self, X, y=None):
        self.sums = np.sum(self.X**2, axis=self.axis, keepdims=True)
        self.scale = np.maximum(np.sqrt(sums), 1e-8)

    def transform(self, X):
        return X / self.scale
