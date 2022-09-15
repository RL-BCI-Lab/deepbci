import deepbci.utils.logger as logger
import tensorflow as tf

class LogToFile(tf.keras.callbacks.Callback):
    """ Keras callback class to output training metrics to log file."""
    def on_epoch_end(self, epoch, logs=None):
        metrics = ' '.join(["{}: {}".format(k, v) for k, v in logs.items()])
        logger.info("Epoch: {} {}".format(epoch, metrics))