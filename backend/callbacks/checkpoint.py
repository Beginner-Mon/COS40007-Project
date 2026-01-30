import torch
from pathlib import Path
from .base import Callback
class ModelCheckpoint(Callback):
    def __init__(self, monitor, dir):
        self.monitor = monitor
        self.best = None
        self.dir = Path(dir)
        self.dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs):
        value = logs[self.monitor]

        if self.best is None or value > self.best:
            self.best = value
            torch.save(logs["model"].state_dict(),
                       self.dir / "best.pt")
