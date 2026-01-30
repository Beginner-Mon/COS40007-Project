from .base import Callback
class ReduceLROnPlateau(Callback):
    def __init__(self, optimizer, monitor, factor, patience, min_lr):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.best = None
        self.counter = 0

    def on_epoch_end(self, epoch, logs):
        value = logs[self.monitor]

        if self.best is None or value < self.best:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for g in self.optimizer.param_groups:
                    g["lr"] = max(g["lr"] * self.factor, self.min_lr)
                self.counter = 0
