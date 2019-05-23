#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import keras.backend as K


def pixel_accuracy(y_true, y_pred):
    acc = K.cast(K.equal(K.argmax(y_true, axis=-1),
                         K.argmax(y_pred, axis=-1)),
                 K.floatx())
    return acc


def dice_coeff(y_true, y_pred, label=0):
    shape = y_pred.get_shape()
    num_classes = shape[-1]
    s = [slice(None, None) for _ in range(len(shape))]
    s[-1] = slice(label, label+1)

    y_pred_one_hot = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=num_classes)
    correct = y_true*y_pred_one_hot
    #true_incorrect = y_true*(1-y_pred_one_hot)
    #pred_incorrect = y_pred_one_hot*(1-y_true)

    correct2 = correct[tuple(s)]
    y_pred_one_hot2 = y_pred_one_hot[tuple(s)]
    y_true2 = y_true[tuple(s)]

    div = K.sum(y_pred_one_hot2 + y_true2, axis=(-2, -3))
    en = 2*K.sum(correct2, axis=(-2, -3))

    delta = 1-K.clip(div, 0, 1)
    div += delta
    en += delta
    dc = en/div
    return dc

