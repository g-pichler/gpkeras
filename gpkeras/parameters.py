#!/usr/bin/env python
# *-* encoding: utf-8 *-*

from datetime import datetime
import logging
from pprint import pformat
from ast import literal_eval
import argparse
import sys

logger = logging.getLogger(__name__)


def get_argument_parser(description='Training Parameters'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, required=True, help="The model to use")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to use")

    parser.add_argument("--trainer", type=str, default="{args.model}Trainer", help="The trainer to use")
    parser.add_argument("--run", type=str, help="Run ID",
                        default="{args.dt:%Y-%m-%dT%H:%M:%S}")

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loglevel", type=str, default="info")
    parser.add_argument("--log_dir", type=str, default="log_dir/{args.run}")

    parser.add_argument("--initial_epoch", type=int, help="Initial epoch",
                        default=0)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs')

    parser.add_argument("--checkpoint_file", type=str,
                        default="checkpoint/{args.run}.{{epoch:04d}}.h5")
    parser.add_argument("--state_file", type=str,
                        default="checkpoint/{args.run}.last.txt")
    parser.add_argument("--load_weights", type=str,
                        default="", help="Load weights from file")
    parser.add_argument("--no-resume", action="store_false", default=True, dest='resume',
                        help="Do not resume from state in state_file")
    parser.add_argument("--checkpoint_period", type=int, help="Period for model checkpoints",
                        default=0)
    parser.add_argument("--checkpoint_best_only", action='store_true', help="Checkpoint only best",
                        default=False)

    parser.add_argument("--best_checkpoint_file", type=str,
                        help="Where to save the checkpoint for the best weights",
                        default="checkpoint/{args.run}.best.h5")
    parser.add_argument("--best_state_file", type=str, help="State file for the best weights",
                        default="checkpoint/{args.run}.best.txt")
    parser.add_argument("--best_checkpoint_target", type=str, help="Target indicator for best checkpoint",
                        default="val_pixel_accuracy")
    parser.add_argument("--best_checkpoint_mode", type=str, help="Determines if a higher or lower value is better",
                        default="min", choices=['min', 'max'])

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument("--max_time", type=int, help="Stop the training after a number of seconds",
                        default=0)
    parser.add_argument("--gpu_memory_fraction", type=float, default=None,
                        help="Set per_process_gpu_memory_fraction for the tensorflow session")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning Rate")
    return parser


class TrainingParameters(object):
    formatted_params = ['run',
                        'trainer',
                        'log_dir',
                        'checkpoint_file',
                        'state_file',
                        'best_checkpoint_file',
                        'best_state_file',
                        'load_weights',
                        ]

    evaluated_params = [
    ]

    params = ['run',
              'model',
              'optimizer',
              'trainer',
              'dataset',
              'log_dir',
              'batch_size',
              'initial_epoch',
              'n_epoch',
              'checkpoint_file',
              'checkpoint_period',
              'checkpoint_best_only',
              'best_checkpoint_file',
              'best_state_file',
              'best_checkpoint_target',
              'best_checkpoint_mode',
              'state_file',
              'resume',
              'max_time',
              'loglevel',
              'gpu_memory_fraction',
              'lr',
              'load_weights',
              ]

    _check_list = list()

    def __init__(self, args: argparse.Namespace):
        self.param_dict = dict()
        for param in self.params:
            self.param_dict[param] = getattr(args, param)

        logging.basicConfig(level=getattr(logging, args.loglevel.upper()),
                            format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
                            )
        self.dt = datetime.now()

        for param in self.formatted_params:
            self.param_dict[param] = self.param_dict[param].format(args=self)

        for param in self.evaluated_params:
            self.param_dict[param] = literal_eval(self.param_dict[param])

        self._check_list.append(self._check_resume)
        self._checks()

    def _checks(self):
        for check in self._check_list:
            check()

    def _check_resume(self):
        if self.resume:
            if not (self.state_file and self.checkpoint_file and self.checkpoint_period > 0):
                logger.error('state_file and checkpoint_file needs to be set with a '
                             'checkpoint_period > 0 for resume to work')
                sys.exit(1)

    def __getattr__(self, item):
        if item in self.params:
            return self.param_dict[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if key in self.params:
            self.param_dict[key] = value
        else:
            object.__setattr__(self, key, value)

    def __getitem__(self, item):
        return self.param_dict[item]

    def __setitem__(self, key, value):
        if key not in self.params:
            raise KeyError('Parameter {} is not defined'.format(key))
        self.param_dict[key] = value

    def __str__(self):
        return pformat(self.param_dict, indent=1)

    def __hash__(self):
        itemlist = [self[param] for param in sorted(self.params)]
        return hash(tuple(itemlist))

    def get(self, key, default=None):
        return self.param_dict.get(key, default=default)
