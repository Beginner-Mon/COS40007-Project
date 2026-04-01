import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def build_loss(loss_cfg, num_classes):
    mode = "multiclass" if num_classes > 2 else "binary"
    if loss_cfg.type == "focal":
        # FocalLoss uses F.cross_entropy internally, which always expects
        # 1-D integer targets — so always use "multiclass" mode.
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0)), "multiclass"
    elif loss_cfg.type == "bce" or (loss_cfg.type == "auto" and num_classes == 2):
        return nn.BCEWithLogitsLoss(), "binary"
    return nn.CrossEntropyLoss(), "multiclass"
