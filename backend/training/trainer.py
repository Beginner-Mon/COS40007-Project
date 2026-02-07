import torch
from typing import Tuple

def _compute_loss_and_acc(
    logits: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    mode: str,
) -> Tuple[torch.Tensor, float]:
    """Shared helper for loss + accuracy (binary or multi-class)."""
    if mode == "binary":
        y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
        loss = criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).view(-1).long()
        acc = (preds == y.view(-1).long()).float().mean().item()
    else:
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()

    return loss, acc


def train_epoch(model, loader, optimizer, criterion, device, grad_clip, mode):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)

        loss, acc = _compute_loss_and_acc(logits, y, criterion, mode)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, mode):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)

        loss, acc = _compute_loss_and_acc(logits, y, criterion, mode)

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)