import torch

def get_device(cfg):
    if cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg
