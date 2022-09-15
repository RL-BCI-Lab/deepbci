from pdb import set_trace

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

import numpy as np
import pdb

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Siamese(Model):
    def __init__(self):
        super(Siamese, self).__init__()

        self.model = {}
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.op = tf.keras.optimizers.Adam(learning_rate=6e-4)

        # Conv2D block
        self.model['l1'] = Conv2D(filters=64, kernel_size=10, strides=1, 
                                  padding='same', activation='relu')
        self.model['l1_mp'] = MaxPool2D(pool_size=2, strides=2)
        self.model['l2'] = Conv2D(filters=128, kernel_size=7, strides=1, 
                                  padding='same', activation='relu')
        self.model['l2_mp'] = MaxPool2D(pool_size=2, strides=2)
        self.model['l3'] = Conv2D(filters=128, kernel_size=4, strides=1, 
                                  padding='same', activation='relu')
        self.model['l3_mp'] = MaxPool2D(pool_size=2, strides=2)
        self.model['l4'] = Conv2D(filters=256, kernel_size=4, strides=1, 
                                  padding='same', activation='relu')
        
        # Dense block
        self.model['l6'] = Flatten()
        self.model['l5'] = Dense(units=4096, activation='sigmoid')

        # Output
        self.model['output'] = Dense(units=1, activation='sigmoid')

    def call(self, x):
        for k, v in self.model.items():
            if k == 'output':
                break
            x = v(x)

        return x
            
    def output(self, x):
        output = Dense(units=1, activation='sigmoid')

        return output(x)

    def distance_measure(self, t, b, type):
        if type == 'l1':
            return tf.abs(t - b)
        elif type == 'l2':
            return tf.math.pow((t - b), 2)
        elif type == 'concat':
            return tf.keras.layers.concatenate([t, b])

    def run_model(self, x_top, x_bottom):
        # Pass pairs through network
        top_vector = self(x_top)
        bottom_vector = self(x_bottom)
        
        # Combine feature vectors from top and bottom networks.
        distance = self.distance_measure(t=top_vector,b=bottom_vector,type='l1')

        # Apply final non-linear output layer
        return self.model['output'](distance)

    @tf.function
    def train(self, x_top, x_bottom, labels):
        with tf.GradientTape() as tape:
            preds = self.run_model(x_top, x_bottom)
            loss = self.loss(labels, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.op.apply_gradients(zip(grads, self.trainable_variables))

        return preds, loss

    @tf.function
    def test(self, x_top, x_bottom, labels):
        preds = self.run_model(x_top, x_bottom)
        loss = self.loss(labels, preds)
        
        return preds, loss