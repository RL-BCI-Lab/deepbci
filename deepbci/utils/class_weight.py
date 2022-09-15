
import sklearn
import numpy as np

from deepbci.utils import logger

def compute_class_weight(y, class_weight='balanced', argmax_axis=1):
    """ Wrapper for Sklearn's compute_class_weights() function 
        
        Calculate class weights based on samples per class.
        
        Args:
            y: Data labels given as 2D or 1D array. If 2D array is given then
                argmax is applied using argmax_axis.
            
            class_weight: Corresponds to class_weight argument in Sklearn's compute_class_weights()
                see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

            argmax_axis: Axis to apply argmax to if y has 2 or more dimensions.
  
  """
    if len(y.shape) > 1:
        y = np.argmax(y, axis=argmax_axis)
        
    classes = np.unique(y)
    
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight, 
        classes=classes, 
        y=y 
    )
    
    cw = {c:w for c, w in zip(classes, class_weights)}
    
    return cw