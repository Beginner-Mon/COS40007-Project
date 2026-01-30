import torch.nn as nn

def build_loss(loss_cfg, num_classes):
    if loss_cfg.type == "bce" or (
        loss_cfg.type == "auto" and num_classes == 2
    ):
        return nn.BCEWithLogitsLoss(), "binary"
    return nn.CrossEntropyLoss(), "multiclass"
