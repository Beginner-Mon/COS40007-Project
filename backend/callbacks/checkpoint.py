import torch
from pathlib import Path
from .base import Callback

class ModelCheckpoint(Callback):
    def __init__(self, monitor: str, save_dir, mode="max"):
        self.monitor = monitor
        self.mode = mode

        self.best = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _is_better(self, value):
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def on_epoch_end(self, epoch, logs):
        value = logs[self.monitor]

        if self._is_better(value):
            self.best = value
            torch.save(
                {
                    "model_state_dict": logs["model"].state_dict(),
                    "best_metric": value,
                    "epoch": epoch,
                },
                self.save_dir / "best.pt"
            )
