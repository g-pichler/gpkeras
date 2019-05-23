#!/usr/bin/env python
# *-* encoding: utf-8 *-*


import keras
import numpy as np
import math
import os
import nibabel as nib
#from scipy.ndimage import zoom
from .util import normalize, get_statistics
from typing import Union, List, Tuple, Iterable, Optional

import random
import logging

logger = logging.getLogger(__name__)


class BatchGenerator(keras.utils.Sequence):
    '''
    Takes a keras Sequence and packages it into batches of given size.

    The last batch is (potentially) shorter.
    '''
    def __init__(self, seq: keras.utils.Sequence, batchsize: int):
        '''

        :param seq: The original sequence
        :param batchsize: Batchsize
        '''
        self.batchsize = batchsize
        self._seq = seq

    @staticmethod
    def _concat_lists_or_arrays(a: Union[List, np.ndarray], b: Union[List, np.ndarray]):
        '''
        Helper function that takes a batch of training samples :a and an additional training sample :b
         and concatenates them along the first dimension, expanding the dimension of :b.

         :b is assumed to be a list of numpy arrays, a list of floats or just a float

        :param a: Batch
        :param b: Additional sample
        :return: Batch and sample concatenated
        '''
        if not isinstance(b, list):
            b = [b]
        b = [np.array([c]) if isinstance(c, float) else c for c in b]
        if not a:
            return [np.expand_dims(c, axis=0) if c.ndim > 1 else c.copy() for c in b]
        else:
            c = a[:]
            for i in range(len(b)):
                c[i] = np.concatenate((a[i], np.expand_dims(b[i], axis=0) if b[i].ndim > 1 else b[i]),
                                      axis=0)
            return c

    def __getitem__(self, item):
        '''
        Returns the requested batch.

        :param item: Number of the batch
        :return: Batch
        '''
        fro, to = item * self.batchsize, min((item + 1) * self.batchsize, len(self._seq))

        data = [[], [], []]
        for i in range(fro, to):
            cur = self._seq[i]
            for j in range(len(cur)):
                data[j] = self._concat_lists_or_arrays(data[j], cur[j])

        return tuple(data)

    def __len__(self):
        '''
        Lenght of the batched sequence. We round up and shorten the last batch if necessary

        :return: length of the sequence
        '''
        return math.ceil(len(self._seq) / self.batchsize)

    def on_epoch_end(self):
        '''
        Forwards the on_epoch_end call to the underlying sequence

        :return: None
        '''
        self._seq.on_epoch_end()


class ProductSequence(keras.utils.Sequence):
    '''
    Produces the product sequence of two or more keras Sequences.
    '''
    def __init__(self,
                 sequences: Iterable[keras.utils.Sequence],
                 shuffle_together: bool = False):
        '''
        The :shuffle_together parameter specifies if the product sequence should shuffle
        on epoch end.

        The on_epoch_end call is forwarded to underlying sequences in any case

        :param sequences: input keras Sequences
        :param shuffle_together: should the product sequence shuffle on epoch end
        '''
        self._sequences = sequences
        self._len = min([len(s) for s in self._sequences])
        self._shuffle_together = shuffle_together
        self._idxs = list(range(self._len))

        self._shuffle()

    def __getitem__(self, item):
        data = [[], [], []]

        for s in self._sequences:
            x0 = s[self._idxs[item]]
            for i in range(len(x0)):
                x1 = x0[i]
                if not isinstance(x1, list):
                    x1 = [x1]
                for x2 in x1:
                    if x2 is not None:
                        data[i].append(x2)
        if not data[2]:
            del data[2]

        return data

    def _shuffle(self):
        if self._shuffle_together:
            random.shuffle(self._idxs)

    def __len__(self):
        return self._len

    def on_epoch_end(self):
        self._shuffle()
        for s in self._sequences:
            s.on_epoch_end()


class NiftiSequence_2D(keras.utils.Sequence):
    '''
    A Keras Sequence that slices Nifti images up into 2D slices
    '''

    def __init__(self,
                 data_list: List[Tuple[str, ...]],
                 label_list: List[Tuple[str, ...]],
                 basedir='.',
                 dim=(128, 128),
                 delta=None,
                 data_transforms=(None,),
                 label_transforms=(None,),
                 weight_function=None,
                 shuffle=True,
                 horizontal_flip=True,
                 vertical_flip=True,
                 normalize=True,
                 avoid_ram=False,
                 expand_channel_dim=True,
                 ):
        '''

        :param data_list: list of lists of data files in basedir
        :param label_list: list of lists of label files in basedir
        :param basedir: basedir for looking for nifti files
        :param dim: tuple of dimension for 2D slices
        :param delta: tuple for 2D shift between slices
        :param data_transforms: tuple of transform functions applied on data
        :param label_transforms: tuple of transform functions applied on labels
        :param weight_function: function to calculate weight of slice from data and label
        :param shuffle: should slices be shuffled on_epoch_end
        :param horizontal_flip: random horizontal flips
        :param vertical_flip: random vertical flips
        :param normalize: normalize data by subtracting mean and dividing by std deviation
        :param avoid_ram: if set to True, apply te transforms in the __getitem__ function
        :param expand_channel_dim: if set to True and the fourth/channel dimension does not exist,
                                   it is expanded.
        '''

        self._transforms = tuple([tuple([t if t is not None else lambda x: x for t in transforms])
                            for transforms in (data_transforms, label_transforms)])

        def get_weight(x, y):
            return [1.0] * len(y) if weight_function is None else weight_function(x, y)

        self._items: List[List[List[np.ndarray]]] = list()
        self._statistics: List[List[Tuple[float, float]]] = list()
        self._shuffle = shuffle
        self._dim = dim
        self._delta = delta if delta is not None else tuple([d // 2 for d in dim])
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip
        self._avoid_ram = avoid_ram
        self._expand_channel_dim = expand_channel_dim
        self._normalize = normalize

        assert len(data_list) == len(label_list)

        for i in range(len(data_list)):
            data_filenames = data_list[i]
            label_filenames = label_list[i]

            files = [[os.path.join(basedir, f) for f in filenames]
                     for filenames in (data_filenames, label_filenames)]

            imgs_nii = [[nib.load(f) for f in files1]
                         for files1 in files]

            imgs_data = [[np.asarray(img_ni.dataobj) for img_ni in files1]
                         for files1 in imgs_nii]

            if self._normalize:
                data_statistics = [get_statistics(d) for d in imgs_data[0]]

            shape = imgs_data[0][0].shape
            slices = shape[2]
            for k, imgs_data1 in enumerate(imgs_data):
                assert all([img_data.shape == shape for img_data in imgs_data1])

            shape1 = shape[:2]
            for i in range(slices):
                for x in range(shape1[0] // self._delta[0]):
                    if x * self._delta[0] + self._dim[0] > shape1[0]:
                        break
                    for y in range(shape1[1] // self._delta[1]):
                        if y * self._delta[1] + self._dim[1] > shape1[1]:
                            break
                        out = list()
                        for k in range(2):
                            imgs_data1 = imgs_data[k]
                            out1 = list()
                            for j in range(len(imgs_data1)):
                                img_data = imgs_data1[j]
                                img_data = img_data[x * self._delta[0]:x * self._delta[0] + self._dim[0],
                                           y * self._delta[1]:y * self._delta[1] + self._dim[1], i]
                                out1.append(img_data)

                            out.append(out1)

                        # Weight
                        out.append(get_weight(out[0], out[1]))

                        if self._avoid_ram:
                            self._items.append(out)
                            if self._normalize:
                                self._statistics.append(data_statistics)
                        else:
                            self._items.append(self._transform_items(out, data_statistics))

        self._idxs = list(range(len(self._items)))

        # Shuffle at the very beginning as well
        self._shuffle_me()

    def __len__(self) -> int:
        '''
        Length of the Sequence
        :return: length
        '''
        return len(self._items)

    def _transform_items(self, items, statistics: List[Tuple[float, float]] = None):
        '''
        Do the nrmalization and apply the transformation function on list of lists of numpy arrays
        :param items: list of lists of numpy arrays
        :param statistics: the statistics of the data, calculated if not given
        :return: transformed list of lists of numpy arrays
        '''
        new_items = items.copy()

        items2 = list()
        for k, img_data in enumerate(items[0]):
            items2.append(normalize(img_data, statistics[k] if statistics is not None else None))
        new_items[0] = items2

        items = new_items

        # Apply transform
        new_items = list()
        for k, out1 in enumerate(items[:2]):
            items2 = list()
            for j, img_data in enumerate(out1):
                items2.append(self._transforms[k][j](img_data))
            new_items.append(items2)

        new_items += items[2:]

        return new_items

    def __getitem__(self, item: int):
        '''
        Gets the item.
        If avoid_ram is enabled the (possible) normalization and transformation also happens here
        :param item: number of the item
        :return: list of lists of numpy arrays
        '''
        items = self._items[self._idxs[item]]
        if self._avoid_ram:
            items = self._items[self._idxs[item]]
            statistics = self._statistics[self._idxs[item]] if self._statistics else None
            items = self._transform_items(items, statistics=statistics)

        if self._horizontal_flip and random.getrandbits(1):
            items = [[np.flip(i, axis=0) for i in items1] for items1 in items[:2]] + items[2:]
        if self._vertical_flip and random.getrandbits(1):
            items = [[np.flip(i, axis=1) for i in items1] for items1 in items[:2]] + items[2:]
        return items

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        '''
        Called at the end of the epoch

        :return: None
        '''
        self._shuffle_me()

    def _shuffle_me(self):
        '''
        Shuffles the slices if shuffling is enabled
        :return: None
        '''
        if self._shuffle:
            random.shuffle(self._idxs)


# if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # c = NiftiSequence_2D(basedir="/home/georg/work/learning-data/da-data/training/supervised",
    #                      )
    #
    # k = 0
    # M = 20
    # for im in c:
    #     if k > M:
    #         plt.imshow(np.squeeze(im[0], -1))
    #         plt.show()
    #     if k > M + 1: break
    #     k += 1
