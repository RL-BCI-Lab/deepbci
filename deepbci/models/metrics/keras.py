import tensorflow as tf
import numpy as np
from sklearn.metrics import  confusion_matrix

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=1), tf.bool)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        self.tp.assign_add(tf.reduce_sum(tf.cast(tp, self.dtype)))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        self.fn.assign_add(tf.reduce_sum(tf.cast(fn, self.dtype)))
        tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tn, self.dtype)))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        self.fp.assign_add(tf.reduce_sum(tf.cast(fp, self.dtype)))

    def result(self):
        e = tf.constant(1e-8)
        tpr = tf.maximum(tf.divide(self.tp, tf.add(self.tp, self.fn)), e)
        tnr = tf.maximum(tf.divide(self.tn, tf.add(self.tn, self.fp)), e)
        
        return tf.divide(tf.add(tpr, tnr), 2)