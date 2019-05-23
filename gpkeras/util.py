from typing import NamedTuple, Dict, Any

from keras import backend as K
import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple, List, Union


def merge_labels_categorical(t: np.ndarray, merge: List[List[int]]):
    '''
    Merge labels together when given in a one hot numpy array.
    :merge is a list indicating what to merge together.
    E.g. merge=[[1,2], [3,0]] would create two labels the first being labels 1 and 2 and the second
    being the merging of 3 and 0.

    :param t: input array
    :param merge: list of lists of label indices
    :return: output array
    '''
    shape = t.shape
    slices = [slice(None, None) for _ in shape]

    l: List[np.ndarray] = list()
    for i, idxs in enumerate(merge):
        for idx in idxs:
            slices[-1] = slice(idx, idx+1)
            my_slice = t[slices]
            if len(l) <= i:
                l.append(my_slice)
            else:
                l[i] += my_slice

    return K.concatenate(l, -1)


def rename_labels(arr: np.ndarray, rename: Dict[int, int]):
    '''
    Renames labels using a dictionary :rename, where rename[fro] = to

    :param arr: input array
    :param rename: dictionary describing the rename operation
    :return: output array
    '''
    arr2 = np.zeros(arr.shape, dtype=np.uint8)
    for fro, to in rename.items():
        arr2[arr == fro] = to
    return arr2


def tensorboard_write_text(text: str, name: str = "Summary", log_dir: str = "./log_dir",
                           global_step: int = 0, writer: Optional[tf.summary.FileWriter] = None):
    '''
    Write a text summary to tensorboard.
    :param text: text to write
    :param name: name of the text summary
    :param log_dir: log_dir to write to
    :param global_step: the global step (epoch) to write at
    :param writer: an optional tensorboard FileWriter class to use
    :return:
    '''
    writer_ = tf.summary.FileWriter(log_dir) if writer is None else writer
    try:
        text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=name, metadata=meta, tensor=text_tensor)
        writer_.add_summary(summary, global_step=global_step)
    finally:
        if writer is None:
            writer_.close()


def get_statistics(x: np.ndarray) -> Tuple[float, float]:
    '''
        Take a numpy array as input and computes its mean and std deviation

        :param x: Input numpy array
        :return: Tuple (mean, std deviation)
    '''
    N = np.prod(x.shape)
    mu = np.sum(x) / N
    sigma = np.sqrt(np.sum((x - mu) ** 2) / (N - 1))

    return mu, sigma


def normalize(x: Union[np.ndarray, np.array], statistics: Optional[Tuple[float, float]] = None):
    '''
    Normalizes an input numpy array by subtracting its mean and dividing by the std deviation
    :param x: input array
    :param statistics: optional statistics, otherwise computed
    :return: centered and normalized array
    '''
    if statistics is None:
        statistics = get_statistics(x)
    mu, sigma = statistics
    return (x-mu)/sigma
