from .base import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor, patience, min_delta=0.0, mode='auto'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False
        
        if mode == 'auto':
            if 'loss' in monitor:
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            self.mode = mode

    def on_epoch_end(self, epoch, logs):
        value = logs[self.monitor]

        if self.best is None:
            self.best = value
            self.counter = 0
        elif self.mode == 'min' and value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        elif self.mode == 'max' and value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            # According to notebook, we only trigger when counter >= patience
            self.counter += 1
            if self.counter >= self.patience:
                print("\nEarly stopping triggered.", flush=True)
                self.stop = True
