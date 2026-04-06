import math
import time
from pathlib import Path

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import compute_metrics, latest_checkpoint


def _append_csv_row(path: Path, row: dict) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def train_feature_classifier(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "sgd",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    early_stopping=True,
                    n_iter_no_change=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def train_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
    ckpt_dir: Path,
    resume: bool,
    save_checkpoints: bool,
    log_csv: Path | None = None,
) -> Path | None:
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = ckpt_dir / "autoencoder_best.pt"
    last_path = ckpt_dir / "autoencoder_last.pt"
    start_epoch = 1
    best_loss = float("inf")

    if resume and save_checkpoints:
        if last_path.exists():
            blob = torch.load(last_path, map_location=device)
            model.load_state_dict(blob["model_state"])
            optimizer.load_state_dict(blob["optimizer_state"])
            start_epoch = int(blob["epoch"]) + 1
            best_loss = float(blob.get("best_loss", best_loss))
            print(f"Resuming AE from: {last_path} (next epoch={start_epoch})")
        elif best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Found AE best checkpoint: {best_path}")

    if start_epoch > epochs and save_checkpoints:
        print(f"AE already trained to epoch {start_epoch-1} (target={epochs}). Skipping training.")
        return best_path if best_path.exists() else (last_path if last_path.exists() else None)

    print("Starting autoencoder training...")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()

        running = 0.0
        n = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            running += loss.item() * bs
            n += bs

        epoch_loss = running / max(n, 1)
        epoch_s = time.time() - t0
        print(f"AE Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f} - time: {epoch_s:.1f}s")

        if log_csv is not None:
            _append_csv_row(
                log_csv,
                {"epoch": epoch, "loss": float(epoch_loss), "seconds": float(epoch_s)},
            )

        if save_checkpoints:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": float(epoch_loss),
                    "best_loss": float(min(best_loss, epoch_loss)),
                },
                last_path,
            )
            print(f"Saved checkpoint (last): {last_path}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_path)
                print(f"New best AE loss: {best_loss:.6f} | Saved: {best_path}")

    return best_path if save_checkpoints else None


@torch.inference_mode()
def eval_baseline_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[int] = []
    probs: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs.extend(p.tolist())
        ys.extend([int(v) for v in y])
    return np.asarray(ys, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def train_baseline_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    ckpt_dir: Path,
    resume: bool,
    log_csv: Path | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_path = ckpt_dir / "baseline_best.pt"
    best_auc = -math.inf

    if resume and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Found CNN best checkpoint: {best_path}")

    print("Starting baseline CNN training...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            running += loss.item() * bs
            n += bs
        train_loss = running / max(n, 1)

        y_true, y_prob = eval_baseline_cnn(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_prob)
        val_auc = float(val_metrics["roc_auc"])
        val_acc = float(val_metrics["accuracy"])

        epoch_s = time.time() - t0
        print(
            f"CNN Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f} - "
            f"val_acc: {val_acc:.6f} - val_auc: {val_auc:.6f} - time: {epoch_s:.1f}s"
        )

        if log_csv is not None:
            _append_csv_row(
                log_csv,
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_acc": float(val_acc),
                    "val_auc": float(val_auc),
                    "seconds": float(epoch_s),
                },
            )

        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"New best CNN AUC: {best_auc:.6f} | Saved: {best_path}")

    return best_path
