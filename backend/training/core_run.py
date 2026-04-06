import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow

from data.motion_dataset import MotionDataset
from model.bilstm import BiLSTM
from model.gru import GRU
from model.tcn import TCN
from training.loss import build_loss
from training.trainer import train_epoch, eval_epoch
from callbacks.early_stopping import EarlyStopping
from callbacks.checkpoint import ModelCheckpoint
from callbacks.lr_scheduler import ReduceLROnPlateau

def run_training_core(X_train, y_train, X_val, y_val, cfg, label_encoder, run_dir, device, fold=1):
    """
    Core function to build models, dataloaders, and run the PyTorch epoch loop.
    Returns the best validation loss, the trained model, and the fitted scaler.
    """
    # Apply scaler safely only on train set
    B_tr, T_tr, F_tr = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F_tr)).reshape(B_tr, T_tr, F_tr)
    
    B_val, T_val, F_val = X_val.shape
    X_val_scaled = scaler.transform(X_val.reshape(-1, F_val)).reshape(B_val, T_val, F_val)

    train_dataset = MotionDataset(X_train_scaled, y_train)
    val_dataset = MotionDataset(X_val_scaled, y_val)

    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, pin_memory=pin_memory)

    num_classes = len(label_encoder.classes_)
    in_size = X_train.shape[2]
    h_size = cfg.model.hidden_size
    n_layers = cfg.model.num_layers
    drop = cfg.model.dropout
    mtype = cfg.model.get("type", "bilstm").lower()

    if mtype == "gru":
        model = GRU(input_size=in_size, hidden_size=h_size, num_classes=num_classes, num_layers=n_layers, dropout=drop).to(device)
    elif mtype == "tcn":
        model = TCN(input_size=in_size, hidden_size=h_size, num_classes=num_classes, num_layers=n_layers, dropout=drop).to(device)
    else:
        model = BiLSTM(input_size=in_size, hidden_size=h_size, num_classes=num_classes, num_layers=n_layers, dropout=drop).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion, mode = build_loss(cfg.loss, len(label_encoder.classes_))

    es, ckpt, rlr = None, None, None
    if cfg.callbacks.early_stopping.enabled:
        es = EarlyStopping(monitor=cfg.callbacks.early_stopping.monitor, patience=cfg.callbacks.early_stopping.patience, min_delta=cfg.callbacks.early_stopping.min_delta)
    if cfg.callbacks.checkpoint.enabled:
        ckpt = ModelCheckpoint(monitor=cfg.callbacks.checkpoint.monitor, save_dir=run_dir, mode="min" if "loss" in cfg.callbacks.checkpoint.monitor else "max")
    if cfg.callbacks.reduce_lr.enabled:
        rlr = ReduceLROnPlateau(optimizer=optimizer, monitor=cfg.callbacks.reduce_lr.monitor, factor=cfg.callbacks.reduce_lr.factor, patience=cfg.callbacks.reduce_lr.patience, min_lr=cfg.callbacks.reduce_lr.min_lr)

    best_val_loss = float('inf')
    for epoch in range(cfg.train.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, cfg.train.grad_clip, mode)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, mode)

        if mlflow.active_run():
            mlflow.log_metrics({
                f"fold_{fold}_train_loss": train_loss, f"fold_{fold}_train_acc": train_acc,
                f"fold_{fold}_val_loss": val_loss, f"fold_{fold}_val_acc": val_acc
            }, step=epoch)

        print(
            f"[Fold {fold}] Epoch {epoch+1}/{cfg.train.epochs}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}",
            flush=True,
        )

        logs = {"loss": train_loss, "acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "model": model}
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if ckpt: ckpt.on_epoch_end(epoch, logs)
        if es:
            es.on_epoch_end(epoch, logs)
            if es.stop:
                print(f"[STOP] Fold {fold} Early stopping!", flush=True)
                break
        if rlr: rlr.on_epoch_end(epoch, logs)
        
    return best_val_loss, model, scaler
