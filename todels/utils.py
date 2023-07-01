import torch

from time import time


class ProgressBar:
    def __init__(self, epochs, n_samples, length_bar):
        self.n_samples = n_samples
        self.epochs = epochs
        self.length_bar = length_bar
        self.reset()
    
    def reset(self):
        self.percent = 0
        self.last_time = time()
        self.take_time = 0
        self._print_result(0)

    def _print_result(self, percent, addition_output="", end=False):
        if end:
            end = '\n'
        else:
            end = ''
        space = self.length_bar - self.percent - 1
        # TODO: get format from user
        print(
            f'\r epochs {self.epochs}: ' \
            f'[{"="*self.percent}>{" "*space}] {percent:>3d}%  ' \
            f'time:{self.take_time:>.3f}s - {addition_output}',
            end=end
        )
        
    def next(self, batch, addition_output=""):
        if batch == self.n_samples:
            self._print_result(100, addition_output, True)
            return
        new_percent = int((batch / self.n_samples) * self.length_bar)
        actual_percent = int((self.percent / self.length_bar) * 100)
        if self.percent < new_percent:
            self.percent = new_percent
        self.take_time += time() - self.last_time
        self.last_time = time()
        self._print_result(actual_percent)


class EarlyStopping:
    def __init__(self, patience, max_tolerant=1):
        self.min_validation = float("inf")
        self.patience = patience
        self.counter = 0
        self.max_tolerant = max_tolerant

    def check(self, validation):
        if validation < self.min_validation:
            self.min_validation = validation
            self.counter = 0
        elif validation > (self.min_validation + self.patience):
            self.counter += 1
            if self.counter >= self.max_tolerant:
                return True
        return False

class Checkpoint():
    def __init__(self, model, path):
        self.path = path
        self.model = model
        self.best = float("-inf")

    def __call__(self, val):
        if self.best < val:
            torch.save(self.model, self.path)

    def load(self):
        return torch.load(self.path)

__all__ = ("ProgressBar", "EarlyStopping", "Checkpoint")