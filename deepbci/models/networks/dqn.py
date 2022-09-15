import pdb
import os
from pdb import set_trace
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class DQN(tf.keras.Model):
    def __init__(self, actions, name='dqn', init_seed=None):
        super(DQN, self).__init__(name=name)
        self.model = {}
        kernel_init = tf.initializers.VarianceScaling(seed=init_seed)
        
        self.model['l1'] = Conv2D(filters=32, 
                                  kernel_size=8, 
                                  strides=(4, 4),
                                  padding='same',  
                                  activation='relu',
                                  kernel_initializer=kernel_init,
                                  name='l1_conv2d')
        self.model['l2'] = Conv2D(filters=64, 
                                  kernel_size=4,
                                  strides=(2, 2),
                                  padding='same', 
                                  activation='relu',
                                  kernel_initializer=kernel_init,
                                  name='l2_conv2d')
        self.model['l3'] = Conv2D(filters=64, 
                                  kernel_size=3,
                                  strides=(1, 1),
                                  padding='same', 
                                  activation='relu',
                                  kernel_initializer=kernel_init,
                                  name='l3_conv2d')
        self.model['l4'] = Flatten(name='l4_flatten')
        self.model['l5'] = Dense(units=512,
                                 activation='relu',
                                 kernel_initializer=kernel_init,
                                 name='l5_dense')
        self.model['output'] = Dense(units=actions,
                                     kernel_initializer=kernel_init,
                                     name='output_dense')
    @tf.function
    def call(self, x):
        for k, v in self.model.items(): 
            x = v(x)
        return x
        
    def init_input(self, shape):
        if len(shape) != 3:
            err = "Invalid input initialization shape {}. There must be 3 dimensions given"
            raise ValueError(err.format(shape))
        
        # Add keras Input
        self(tf.keras.Input(shape=shape, name='input'))
        
        # Pass dummy tensor through model
        input_shape = [1] + list(shape)
        init_tensor = tf.zeros(input_shape)
        self(init_tensor)