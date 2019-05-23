#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import tensorflow as tf
import keras.backend as K
import keras.losses
import numpy as np


def spherical_loss(y_true, y_pred):
    norm = tf.norm(y_pred, axis=-1, keepdims=True)
    length = tf.norm(y_pred, ord=1, axis=-1, keepdims=True)
    loss = (y_true * y_pred) / (norm * length)
    loss = 1 - tf.reduce_sum(loss, axis=1)
    return loss


def brier_loss(y_true, y_pred):
    loss = tf.reduce_sum(2 * y_pred * y_true, axis=-1)
    sq_norm = tf.reduce_sum(K.square(y_pred), axis=-1)
    loss = 1 - loss + sq_norm
    return loss


# Ignore y_true, just take y_pred as loss
def prediction_loss(y_true, y_pred):
    return y_pred


# Return all zeros
def constant_zero(y_true, y_pred):
    return K.zeros_like(y_true)


# Hellinger distance
def squared_hellinger_distance(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)
    return K.sum(
        y_true + y_pred
        - 2 * K.sqrt(y_pred) * K.sqrt(y_true),
        axis=-1)


# Bhattacharyya distance
def bhattacharyya_distance(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)
    return -K.log(K.sum(K.sqrt(y_true * y_pred), axis=-1))


# Bhattacharyya distance
def negative_bhattacharyya_kernel(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)
    return 1 - (K.sum(K.sqrt(y_true * y_pred), axis=-1))


# Weighted categorical_crossentropy
def gen_weighted_categorical_crossentropy(weights, n_dims):
    for _ in range(n_dims):
        weights = np.expand_dims(weights, 0)
    weight_tensor = K.variable(weights)

    def weighted_categorical_crossentropy(y_true, y_pred):
        return keras.losses.categorical_crossentropy(y_true * weight_tensor, y_pred)

    return weighted_categorical_crossentropy
