import os
import traceback
from typing import Tuple, List
from pdb import set_trace
from os.path import join

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import deepbci as dbci
from deepbci.utils import logger
from deepbci.utils.utils import tf_allow_growth
from deepbci.models.wrappers.base import BaseWrapper, BaseDataset
        
class KerasDataset(BaseDataset):
    def __init__(
        self, 
        trn_set=None, 
        vld_set=None, 
        tst_set=None, 
        *,
        shuffle=False,
        seed=None,
        batch_size=128, 
        buffer_size=None
    ):
        """ Keras dataset wrapper.
        
            Attributes:
                            
                batch_size (int): Size of each batch.
                
                buffer_size (int): Determines the size of data buffer which is then used to
                    randomly select samples. If shuffle is False then this argument is not
                    used. If this argument is None then it is set to the size of your data.Data.
        """
        super().__init__(trn_set, vld_set, tst_set, seed=seed, shuffle=shuffle)
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    @BaseDataset.trn_set.getter
    def trn_set(self):
        if self._trn_set:
            return self.build_dataset(*self._trn_set, shuffle=self.shuffle)
        else:
            return None

    @BaseDataset.vld_set.getter
    def vld_set(self):
        if self._vld_set:
            return self.build_dataset(*self._vld_set)
        else:
            return None
        
    @BaseDataset.tst_set.getter
    def tst_set(self):
        if self._tst_set:
            return self.build_dataset(*self._tst_set)
        else:
            return None
        
    def build_dataset(self, data, labels=None, shuffle=False):
        """ Build and prepare data using tf.data.Dataset.
            
            See tensorflow tf.data.Dataset.shuffle for more information can be found 
            here https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle.
            
            WARNING:
                Bug with TensorFlow 2.0.0 where pairing tf.data.dataset with the
                class_weight parameter for Model.fit() causes fit() method to ignore
                class_weight parameter. DO NOT USE IF YOU ARE USING class_weight! 
                Reference: https://github.com/tensorflow/tensorflow/issues/33550  
            
            Args:
                data (np.ndarray): Data given as a NumPy array.
                
                labels (np.ndarray): Labels given as a NumPy array.
                
                shuffle (bool): Determines if data should be shuffled.
        """
        buffer_size = self.buffer_size if self.buffer_size else len(data)

        # Create TensorFlow Dataset and cast data and labels as float32
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (data.astype(np.float32), labels.astype(np.float32))
        )

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size, seed=self.seed)
            
        tf_dataset = tf_dataset.batch(self.batch_size)

        return tf_dataset

class Keras(BaseWrapper):
    """ Base wrapper for Keras model for training and testing.

        All TensoFlow models should implement this class and override the fit and test methods
        if special training and testing processing is needed. For basic models this wrapper
        should suffice and is technically redundant. 

        Although redundant the model wrapper classes should allow for easier plug an play 
        when using models with unique training and testing processing, e.g. meta learning.
    """
    _dataset = KerasDataset
    
    def __init__(
        self, 
        model=None,  
        loss=None, 
        metrics=None, 
        callbacks=None,
        optimizer='Adam',
        load:dict=None
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics if metrics else []
        self.callbacks = callbacks if callbacks else []

        if self.model:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif load:
            self.load(**load)
        else:
            err = "Neither model nor load arguments were passed to initialize the Keras wrapper model."
            raise ValueError(err)
        
    @staticmethod
    def preinstantiate(allow_growth=True):
        if allow_growth:
            try:
                logger.info(f"Attempting to enable TensorFlow's memory growth...")
                tf_allow_growth()
            except Exception as e:
                logger.info(f"Failed to set TensorFlow memory growth: {e}")
        logger.log(f"Memory growth enabled!")
        tf.keras.backend.clear_session()
    
    def fit(
        self, 
        trn_set: Tuple[tf.data.Dataset], 
        vld_set: Tuple[tf.data.Dataset] = None, 
        preload_checkpoint: str = None,
        **kwargs
    ):
        """ Fit TensorFlow model using Keras trainer."""
        if preload_checkpoint:
            logger.info(f"Attempting to load checkpoint from {preload_checkpoint}...")
            self.load(preload_checkpoint)
        
        hist = self.model.fit(
            trn_set, 
            validation_data=vld_set, 
            callbacks=self.callbacks,
            **kwargs
        )
        
        self.log_summary()
        
        model_ckpt = self._has_checkpoint()
        if model_ckpt is not None:
            logger.info(f"Attmepting to load best weights from training at {model_ckpt.filepath}...")
            if os.path.exists(model_ckpt.filepath):
                self.load(model_ckpt.filepath)
            else:
                logger.warn(f"Checkpoint file path does not exists: {model_ckpt.filepath}")

    def _has_checkpoint(self):
        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                return cb
        return None
    
    def predict(self, tst_set, **kwargs):
        """ Tests model based on current weights.
        
            This method runs tf.keras.model.predict() to get a models predictions and 
            tf.keras.model.evalute() to get models metrics.
            
            Returns:
                Dictionary of metrics 
        """
        return self.model.predict(x=tst_set, **kwargs) 
    
    def save(self, filepath: str, weights_only: bool = True, **kwargs):
        """ Saves either just the weights or entire model.
        
            See the "Save the entire model" section within TensorFlow's official docs: 
            https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model
        """
        self._make_dir(filepath)
        if weights_only:
            self.model.save_weights(filepath, **kwargs)
        else:
            self.model.save(filepath, **kwargs)

    def load(self, filepath: str, weights_only: bool = True, **kwargs):
        """ Loads either just the weights or the entire model"""
        if weights_only:
            self.model.load_weights(filepath, **kwargs).expect_partial()
        else:
            self.model = tf.keras.models.load_model(filepath, **kwargs)
    
    def log_summary(self):
        """ Logs tf.keras.model summary information """
        self.model.summary(print_fn=logger.info)

    def get_layer_names(self):
        """ Get the custom and class name for each layer in the model.

            Return:
                List of custom names and a list of the class names. 
        """
        custom_names = []
        class_names = []
        for l in self.model.layers:
            custom_names.append(l.name)
            class_names.append(l.__class__.__name__)
        
        return custom_names, class_names

    def get_layer_outputs(self, dataset, inputs):
        """ Gets the output of each layer in the model.

            Returns:
                List of np.ndarrays contianing output of each layer.

        """
        output_refs = [l.output for l in self.model.layers]
        # Create a new model where all layers give their outputs
        new_model = tf.keras.Model(inputs, output_refs)
        layer_outputs = new_model.predict(dataset)
        return layer_outputs