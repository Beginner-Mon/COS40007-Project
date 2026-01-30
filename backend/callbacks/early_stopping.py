from .base import Callback
class EarlyStopping(Callback):
    def __init__(self, monitor, patience, min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def on_epoch_end(self, epoch, logs):
        value = logs[self.monitor]

        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
