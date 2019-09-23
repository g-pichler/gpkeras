#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import os
import json
from typing import Dict, Any
import logging
import os.path as osp
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, generate_image_fkt, tags=("Image"), log_dir="./log_dir"):
        super().__init__()
        self._tags = tags
        self._generate_img_fkt = generate_image_fkt
        self._log_dir = log_dir

    def _make_image(self, tensor, black_white=False):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """

        black_white = len(tensor.shape) > 2

        if black_white:
            mode = None
        else:
            mode = "L"

        from PIL import Image
        if black_white:
            height, width, channel = tensor.shape
        else:
            height, width = tensor.shape
            channel = 1
        image = Image.fromarray(tensor, mode=mode)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        imgs = self._generate_img_fkt()

        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        for tag, img in zip(self._tags, imgs):
            image = self._make_image(img)
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
            writer = tf.summary.FileWriter(self._log_dir)
            writer.add_summary(summary, epoch)
            writer.close()

        return


class Sleeper(keras.callbacks.Callback):
    _sleeptime: float

    def __init__(self,
                 sleeptime: float):
        super(Sleeper, self).__init__()
        self._sleeptime = sleeptime
        import warnings
        warnings.filterwarnings(action='ignore',
                                category=UserWarning,
                                message="Method on_batch_end\\(\\) is slow compared to the batch update")

    def on_batch_end(self, batch, logs=None):
        time.sleep(self._sleeptime)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    @staticmethod
    def _flatten(data):
        out = []
        for x in data:
            out += x
        return out

    def __init__(self, batch_gen=None, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.

        if self.batch_gen is not None:
            data = self.batch_gen[0]
            val_bsize = data[0][0].shape[0]
            val_len = len(data[0])

            if len(data) == 3:
                self.validation_data = self._flatten(data)
            else:
                self.validation_data = self._flatten(data + [[np.ones(val_bsize)] * val_len])

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    last_epoch: Optional[int] = None
    last_logs: Optional[Dict[str, Any]] = None

    def __init__(self, *args, state_file=None, **kwargs, ):
        super(ModelCheckpoint, self).__init__(*args, **kwargs)
        directory = osp.dirname(self.filepath)
        os.makedirs(directory, exist_ok=True)
        self.state_file = state_file

    def on_epoch_end(self, epoch, logs=None):
        super(ModelCheckpoint, self).on_epoch_end(epoch=epoch, logs=logs)
        self.last_epoch = epoch
        self.last_logs = logs
        current = logs.get(self.monitor)
        if (not self.save_best_only) or self.best == current:
            self._update_state()

    def write_current_and_update_state(self):
        if self.last_epoch is None:
            logger.warning("Cannot save state: no epoch completed")
            return
        filepath = self.filepath.format(epoch=self.last_epoch + 1, **self.last_logs)
        logger.info("Saving model checkpoint to \"{}\"".format(filepath))
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)
        self._update_state()

    def _update_state(self):
        if self.state_file:
            filepath = self.filepath.format(epoch=self.last_epoch + 1, **self.last_logs)
            relative_filepath = osp.relpath(filepath,
                                            osp.dirname(self.state_file))
            if os.path.exists(filepath):
                with open(self.state_file, "w") as fp:
                    json.dump((self.last_epoch + 1, relative_filepath), fp)
