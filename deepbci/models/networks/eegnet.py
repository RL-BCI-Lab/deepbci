"""
@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}

https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
https://github.com/vlawhern/arl-eegmodels/blob/master/examples/ERP.py
"""

from pdb import set_trace

from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import numpy as np

class EEGNet(tf.keras.Model):
    """ EGGNet implementation. Original paper implementation can be found here:
        https://arxiv.org/abs/1611.08024. We have kept the authors original
        details about the algorithm in the comments. However, we have renamed
        some parameters for ease understanding.
          
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
        While the original paper used Dropout, we found that SpatialDropout2D 
        sometimes produced slightly better results for classification of ERP 
        signals. However, SpatialDropout2D significantly reduced performance 
        on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
        the default Dropout in most cases.
            
        Assumes the input signal is sampled at 128Hz. If you want to use this model
        for any other sampling rate you will need to modify the lengths of temporal
        kernels and average pooling size in blocks 1 and 2 as needed (double the 
        kernel lengths for double the sampling rate, etc). Note that we haven't 
        tested the model performance with this rule so this may not work well. 
        
        The model with default parameters gives the EEGNet-8,2 model as discussed
        in the paper. This model should do pretty well in general, although it is
        advised to do some model searching to get optimal performance on your
        particular dataset.
        
        We set separable_filters = conv_filters * depthwise_filters (number of 
        input filters = number of output filters) for the SeparableConv2D layer. 
        We haven't extensively tested the values of this parameter (say, 
        separable_filters < conv_filters * depthwise_filters for compressed learning, 
        and separable_filters > conv_filters * depthwise_filters for overcomplete). 
        We believe the main parameters to focus on are conv_filters and depthwise_filters. 
            
        Args:
            window_size (int): Time points in the EEG data
            
            chs (int): Number of channels or electrodes used in the EEG data
            
            windows (int): Number of epoched windows for a given data samples.
                This dimension acts as the filter dimension as well.
            
            classes (int): Number of classes being predicted.
            
            conv_kern_len (int): Represents the kernel length for the basic
                convolution layer. This value is typically set to half the value of fs.
            
            conv_filters (int): Represents the number of filters for the basic
                convolution layer. Referred to as F1 in EEGNet paper.
                
            depthwise_filters (int): Represents a multiplier to determine number 
                of filters used for each existing filter in the depthwise layer. 
                Meaning, there are x many unique filters applied to each existing 
                filter. Referred to as D in EEGNet paper.
            
            separable_filters (int): Represents the number of pointwise filters
                to apply in the separable convolution layer. This number is typically
                equal to number of filters output in the depthwise layer. Referred to 
                as F2 in EEGNet paper.
                
                Default:
                    separable_filters = conv_filters * depthwise_filters
                    
                Theory:
                    If separable_filters < conv_filters * depthwise_filters then
                    compression is applied to the representations.
                    
                    If separable_filters > conv_filters * depthwise_filters the
                    representations are overcomplete.
                    
            norm_rate (float): Max normalization rate for dense layer.
            
            drop_rate (float): Dropout rate for the model.
                    
            drop_type (str): Should be set to either Either SpatialDropout2D or 
                Dropout to select the type of dropout used.
                
            subsample (list): List of rates to subsample the window size by. These
                values are used in first and second average pooling layers.
        """
    
    def __init__(self, chs, n_classes,
                 fs=128,
                 windows=1,
                 conv_kern_len=64, 
                 conv_filters=8, 
                 depthwise_filters=2, 
                 separable_filters=16,
                 separable_kern=16,
                 norm_rate=0.25,
                 drop_rate=0.5,  
                 drop_type='Dropout',
                 avg_kern_b1=4,
                 avg_kern_b2=8,
                 **kwargs):
        super(EEGNet, self).__init__(name='eegnet', **kwargs)
     
        drop_type = self._select_drop_type(drop_type)
        
        # Block 1
        self.b1_conv2d = Conv2D(filters=conv_filters, 
                                kernel_size=(windows, conv_kern_len), 
                                padding='same',
                                use_bias=False,
                                data_format='channels_first',
                                name='b1_conv2d')
        self.b1_norm = BatchNormalization(axis=1, name='b1_batchnorm')
        self.b1_depth = DepthwiseConv2D(kernel_size=(chs, 1), 
                                        use_bias=False, 
                                        depth_multiplier=depthwise_filters,
                                        depthwise_constraint=max_norm(1.),
                                        data_format='channels_first',
                                        name='b1_depthwise')
        self.b1_norm2 = BatchNormalization(axis=1, name='b1_batchnorm_2')
        self.b1_activation = Activation('elu', name='b1_activation')
        # Default = 128 // 4 = 32 Hz
        self.b1_pool = AveragePooling2D((1, avg_kern_b1),
                                        data_format='channels_first',
                                        name='b1_avg_pool')
        self.b1_dropout = drop_type(drop_rate, name='b1_dropout')
        
        # Block 2
        self.b2_separable = SeparableConv2D(filters=separable_filters, 
                                            kernel_size=(1, separable_kern),
                                            use_bias=False, 
                                            padding='same',
                                            data_format='channels_first',
                                            name='b2_separable')
        self.b2_norm = BatchNormalization(axis=1, name='b2_batchnorm')
        self.b2_activation = Activation('elu', name='b2_activation')
        # Default = 128 // 4 // 8 = 4 Hz
        self.b2_pool = AveragePooling2D((1, avg_kern_b2),
                                        data_format='channels_first',
                                        name='b2_avg_pool')
        self.b2_dropout = drop_type(drop_rate, name='b2_dropout')
        
        # Block 3
        self.b3_flatten = Flatten(name='flatten')
        self.b3_dense  = Dense(units=n_classes, 
                               name='b3_dense', 
                               kernel_constraint=max_norm(norm_rate))
        self.b3_activation = Activation('softmax', name='softmax')

    def call(self, x, training=False):
        # Block 1
        x = self.b1_conv2d(x)
        x = self.b1_norm(x, training=training)
        x = self.b1_depth(x)
        x = self.b1_norm2(x, training=training)
        x = self.b1_activation(x)
        x = self.b1_pool(x)
        x = self.b1_dropout(x, training=training)
        
        # Block 2
        x = self.b2_separable(x)
        x = self.b2_norm(x, training=training)
        x = self.b2_activation(x)
        x = self.b2_pool(x)
        x = self.b2_dropout(x, training=training)
        
        # Block 3
        x = self.b3_flatten(x)
        x = self.b3_dense(x)
        x = self.b3_activation(x)

        return x
    
    def init_input(self, shape):
        if len(shape) != 3:
            err = "Invalid input initialization shape {}. There must be 3 dimensions given"
            raise ValueError(err.format(shape))
        
        # Add keras Input
        inputs = tf.keras.Input(shape=shape, name='input')
        self.call(inputs)
        
        return inputs
        
    def _select_drop_type(self, drop_type):
        if drop_type.lower() == 'spatialdropout2D':
            return SpatialDropout2D
        elif drop_type.lower() == 'dropout':
            return Dropout
        else:
            raise ValueError('drop_type must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')