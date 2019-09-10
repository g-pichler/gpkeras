#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import numpy as np
import keras.utils
import nibabel as nib
from typing import List, Tuple
from .preprocessing import change_labels
from keras.utils import to_categorical
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def get_label_frequencies(generator):
    freq = None
    for item in generator:
        labels = item[1][0]
        s = np.sum(labels, axis=(0, 1))
        freq = freq + s if freq is not None else s
    freq = freq.astype(dtype=np.float)
    freq /= np.sum(freq)
    return freq


def get_label_frequencies_nii(label_list: List,
                              label_merging: Tuple = None,
                              n_labels: int = None):
    assert label_merging is not None or n_labels is not None

    if label_merging:
        n_labels = len(label_merging)
    freq = np.zeros((n_labels,), dtype=np.uint64)
    total = 0
    for i, lfile in enumerate(label_list):
        print("{}/{}: {}...".format(i + 1, len(label_list), lfile))
        label_tensor = np.asarray(nib.load(lfile).dataobj)
        if label_merging:
            label_tensor = change_labels(label_tensor, label_merging)
        lbls, counts = np.unique(label_tensor, return_counts=True)
        for k, l in enumerate(lbls):
            freq[l] += counts[k]
        total += np.prod(label_tensor.shape)
    return np.reshape(freq/total, (n_labels,))

def get_label_frequencies_img(label_list: List,
                              label_merging: Tuple = None,
                              n_labels: int = None):
    assert label_merging is not None or n_labels is not None

    if label_merging:
        n_labels = len(label_merging)
    freq = np.zeros((n_labels,), dtype=np.uint64)
    total = 0
    for i, lfile in enumerate(label_list):
        print("{}/{}: {}...".format(i + 1, len(label_list), lfile))
        label_tensor = np.array(Image.open(lfile))
        if label_merging:
            label_tensor = change_labels(label_tensor, label_merging)
        lbls, counts = np.unique(label_tensor, return_counts=True)
        for k, l in enumerate(lbls):
            freq[l] += counts[k]
        total += np.prod(label_tensor.shape)
    return np.reshape(freq/total, (n_labels,))


def dice_coeff(y_true: np.ndarray, y_pred: np.ndarray):
    num_classes = y_pred.shape[-1]
    #y_true_cat = keras.utils.to_categorical(y_true, num_classes=num_classes)
    # y_true_cat = y_true
    #y_pred_cat = keras.utils.to_categorical(np.argmax(y_pred, axis=-1), num_classes=num_classes)
    y_true_val = np.squeeze(y_true.astype(dtype=np.uint8))
    y_pred_val = np.argmax(y_pred, axis=-1).astype(dtype=np.uint8)
    correct = (y_true_val == y_pred_val).astype(np.uint8)
    correct = correct * (y_true_val + 1) - 1
    val, cnt = np.unique(y_true_val, return_counts=True)
    true_counts = np.zeros((num_classes,), dtype=np.uint64)
    for v, c in zip(val, cnt):
        true_counts[v] = c
    val, cnt = np.unique(y_pred_val, return_counts=True)
    pred_counts = np.zeros((num_classes,), dtype=np.uint64)
    for v, c in zip(val, cnt):
        pred_counts[v] = c
    val, cnt = np.unique(correct, return_counts=True)
    correct_counts = np.zeros((num_classes,), dtype=np.uint64)
    for v, c in zip(val, cnt):
        if v < num_classes:
            correct_counts[v] = c

    return 2 * correct_counts / (pred_counts + true_counts)

