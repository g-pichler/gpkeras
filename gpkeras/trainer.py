#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from keras.backend.tensorflow_backend import set_session
from typing import Callable, Tuple, Dict, List, Any
import enum
from collections import defaultdict

from .parameters import TrainingParameters
from gpkeras import callbacks
from .util import tensorboard_write_text
import keras
import logging
import re
import json
import sys
import signal
import time
from gpkeras.callbacks import TensorBoardWrapper
import threading
import os
import os.path as osp
import tensorflow as tf
from pathlib import Path

Stages = enum.Enum("Stages", ("checks",
                              "backend",
                              "model",
                              "data",
                              "optimizer",
                              "callbacks",
                              "compile",
                              "custom",
                              "training",
                              "cleanup",  # special
                              )
                   )

logger = logging.getLogger(__name__)


class Trainer:
    _args: TrainingParameters
    _stage_callbacks: Dict[Stages, List[Tuple[Callable, bool]]] = defaultdict(list)

    def __init__(self, args: TrainingParameters):
        self._args = args

    def register(self, fkt: Callable, stage: Stages, external: bool = False, first: bool = False):
        if first:
            self._stage_callbacks[stage].insert(0, (fkt, external))
        else:
            self._stage_callbacks[stage].append((fkt, external))

    def unregister(self, fkt: Callable, stage: Stages = None) -> bool:
        stages: Tuple[Stages, ...] = tuple(Stages) if stage is None else (stage,)
        for stage in stages:
            for k, itm in enumerate(self._stage_callbacks[stage]):
                if itm[0] == fkt:
                    del self._stage_callbacks[stage][k]
                    return True
        return False

    def _execute_stage(self, stage: Stages) -> None:
        for fkt, external in self._stage_callbacks[stage]:
            if external:
                ret = fkt(self)
            else:
                ret = fkt()
            if ret is False:
                raise RuntimeError("Function {!s} failed at stage {!s}".format(fkt, stage))

    def __call__(self, *args, **kwargs):

        def exit_handler(_signo, _stack_frame):
            # Raises SystemExit(99):
            logger.error(f'We received signal {_signo}. Exiting...')
            sys.exit(99)

        # We want to die gracefully
        signal.signal(signal.SIGTERM, exit_handler)
        signal.signal(signal.SIGHUP, exit_handler)
        signal.signal(signal.SIGINT, exit_handler)

        for stage in Stages:
            logger.debug("Starting stage \"{stage.name}\"".format(stage=stage))
            try:
                self._execute_stage(stage)
            except (SystemExit, Exception):
                logger.error('An exception occured. Running cleanup stage...')
                self._execute_stage(Stages.cleanup)
                raise


class KerasTrainer(Trainer):
    model: keras.Model = None
    optimizer: keras.optimizers.Optimizer = None
    keras_callbacks: List[keras.callbacks.Callback] = list()
    _modelcheckpoint: callbacks.ModelCheckpoint
    _tensorboard: callbacks.TensorBoard = None
    dataset: Any = None
    model_namespace = keras.models
    optimizer_namespace = keras.optimizers
    dataset_namespace = keras.datasets
    _lock_file: Path

    def __init__(self, args: TrainingParameters):
        super(KerasTrainer, self).__init__(args=args)
        self.register(self.check_number_of_epochs, Stages.checks)
        self.register(self.check_lock, Stages.checks)
        self.register(self.setup_keras, Stages.backend)
        self.register(self.setup_model, Stages.model)
        self.register(self.load_weights, Stages.model)
        self.register(self.setup_optimizer, Stages.optimizer)
        self.register(self.setup_data, Stages.data)
        self.register(self.tensorboard_info, Stages.custom)
        self.register(self.setup_max_time, Stages.custom)
        self.register(self.setup_callback_tensorboard, Stages.callbacks)
        self.register(self.setup_callback_modelcheckpoint, Stages.callbacks)
        self.register(self.setup_callback_best_checkpoint, Stages.callbacks)
        self.register(self.setup_compile, Stages.compile)

    def check_lock(self):
        self._lock_file = Path(osp.join(self._args.log_dir, 'lock'))
        os.makedirs(self._args.log_dir, exist_ok=True)
        if self._lock_file.exists():
            logger.error('Lock file {!s} exists. '
                         'Delete if no training is running.'.format(self._lock_file))
            sys.exit(1)
        else:
            self._lock_file.touch()
            self.register(self.cleanup_lock, Stages.cleanup)

    def cleanup_lock(self):
        if not self._lock_file.exists():
            logger.warning('Lock file {!s} not present. '
                           'Did someone else delete it?'.format(self._lock_file))
        else:
            self._lock_file.unlink()

    def check_number_of_epochs(self):
        if self._args.n_epoch <= self._args.initial_epoch:
            logger.warning("inital_epoch larger or equal to n_epoch; nothing to do")
            logger.warning("exiting ...")
            sys.exit(0)

    def setup_max_time(self):
        def goodbye():
            logger.error("Timeout reached. Goodbye!")
            os.kill(os.getpid(), signal.SIGTERM)

        if self._args.max_time > 0:
            t = threading.Timer(interval=self._args.max_time,
                                function=goodbye)
            t.daemon = True
            logger.info('Starting timer for {!s} seconds'.format(self._args.max_time))
            t.start()

    def setup_data(self):
        d = getattr(self.dataset_namespace, self._args.dataset)
        self.dataset = d(self._args, self.model)

    def save_checkpoint(self):
        if self._modelcheckpoint:
            self._modelcheckpoint.write_current_and_update_state()

    def setup_keras(self):
        if self._args.gpu_memory_fraction is not None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self._args.gpu_memory_fraction
            set_session(tf.Session(config=config))

    def load_weights(self):
        if self._args.resume:
            assert self._args.state_file
            try:
                with open(self._args.state_file, "r") as fp:
                    self._args.initial_epoch, self._args.load_weights = json.load(fp)
                    self._args.load_weights = osp.join(osp.dirname(self._args.state_file),
                                                       osp.basename(self._args.load_weights))
            except FileNotFoundError:
                logger.warning("State file {} not found, not loading old state".format(self._args.state_file))
            else:
                logger.info("Resuming epoch {} from {}".format(self._args.initial_epoch, self._args.load_weights))
        if self._args.load_weights:
            logger.info("Loading weights from {}".format(self._args.load_weights))
            self.model.load_weights(self._args.load_weights)

    def setup_optimizer(self):
        o = getattr(self.optimizer_namespace, self._args.optimizer)
        self.optimizer = o(lr=self._args.lr)

    def setup_model(self):
        m = getattr(self.model_namespace, self._args.model)
        self.model = m(self._args)

    def setup_callback_tensorboard(self):
        if self._args.log_dir:
            self._tensorboard = TensorBoardWrapper(log_dir=self._args.log_dir,
                                                   batch_size=self._args.batch_size,
                                                   )
            self.keras_callbacks.append(self._tensorboard)

    def setup_callback_modelcheckpoint(self):
        if self._args.checkpoint_file and self._args.checkpoint_period > 0:
            self._modelcheckpoint = callbacks.ModelCheckpoint(filepath=self._args.checkpoint_file,
                                                              state_file=self._args.state_file,
                                                              period=self._args.checkpoint_period,
                                                              save_best_only=self._args.checkpoint_best_only,
                                                              )
            self.keras_callbacks.append(self._modelcheckpoint)

    def setup_callback_best_checkpoint(self):
        if self._args.best_checkpoint_file:
            best_checkpoint = callbacks.ModelCheckpoint(filepath=self._args.best_checkpoint_file,
                                                        state_file=self._args.best_state_file,
                                                        period=1,
                                                        save_best_only=True,
                                                        monitor=self._args.best_checkpoint_target,
                                                        mode=self._args.best_checkpoint_mode,
                                                        )
            self.keras_callbacks.append(best_checkpoint)

    def setup_compile(self):
        self.model.compile(self.optimizer)

        logger.info("Model Summary:")
        self.model.summary(print_fn=logger.info)

    def tensorboard_info(self):
        if self._tensorboard:
            tensorboard_write_text(text=re.sub("^", "\t",
                                               "Arguments: \n{}".format(
                                                   str(self._args))
                                               , flags=re.M),
                                   name="Arguments",
                                   log_dir=self._args.log_dir,
                                   global_step=self._args.initial_epoch,
                                   )
            # Ugly hack to make sure we get our own TB file
            logger.debug("Sleeping for 2 seconds")
            time.sleep(2)
