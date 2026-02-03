import torch

def train_epoch(model, loader, optimizer, criterion, device, grad_clip, mode):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)

        if mode == "binary":
            y = y.float().unsqueeze(1)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).view(-1).int()
            acc = (preds == y.view(-1).int()).float().mean()
        else:
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / len(loader), total_acc / len(loader)
