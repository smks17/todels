import torch

from time import time


class ProgressBar:
    def __init__(self, n_samples, max_epochs, lenght_bar):
        self.reset()
        self.n_samples = n_samples
        self.max_epochs = max_epochs
        self.lenght_bar = lenght_bar
        
    def reset(self):
        self.percent = 0
        self.last_time = time()
        self.take_time = 0

    def _print_result(self, percent, addition_output, end=False):
        if end:
            end = '\n'
        else:
            end = ''
        space = self.lenght_bar - self.percent - 1
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
        new_percent = int((batch / self.n_samples) * self.lenght_bar)
        actual_percent = int((self.percent / self.lenght_bar) * 100)
        if self.percent < new_percent:
            self.percent = new_percent
        self.take_time += time() - self.last_time
        self.last_time = time()
        self._print_result(actual_percent, addition_output)


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