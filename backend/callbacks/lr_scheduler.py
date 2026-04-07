from .base import Callback

class ReduceLROnPlateau(Callback):
    def __init__(self, optimizer, monitor, factor, patience, min_lr, mode='auto', min_delta=1e-4):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience # PyTorch patience allows 'patience' bad epochs before reducing on the (patience+1)th
        self.min_lr = min_lr
        self.min_delta = min_delta

        self.best = None
        self.counter = 0
        
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
            self.counter += 1
            if self.counter > self.patience:
                old_lr = self.optimizer.param_groups[0]["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if new_lr < old_lr:
                    print(f"[lr] {old_lr:.6f} -> {new_lr:.6f}", flush=True)
                    
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                self.counter = 0
