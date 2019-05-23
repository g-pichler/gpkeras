from typing import List
import argparse

class TrainingParameters:
    params: List[str]

    run: str
    model: str
    optimizer: str
    trainer: str
    dataset: str
    log_dir: str
    batch_size: int
    initial_epoch: int
    n_epoch: int
    checkpoint_file: str
    checkpoint_period: int
    checkpoint_best_only: bool
    best_checkpoint_file: str
    best_state_file: str
    best_checkpoint_target: str
    best_checkpoint_mode: str
    state_file: str
    resume: bool
    max_time: int
    loglevel: str
    gpu_memory_fraction: float
    lr: float
    load_weights: str

    def __init__(self, args: argparse.Namespace):
        pass

    def _checks(self):
        pass

    def _check_resume(self):
        pass

    def __getattr__(self, item):
        pass

    def __setattr__(self, key, value):
        pass

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def __str__(self):
        pass

    def __hash__(self):
        pass

    def get(self, key, default=None):
        pass


def get_argument_parser(description: str = None) -> argparse.ArgumentParser:
    pass
