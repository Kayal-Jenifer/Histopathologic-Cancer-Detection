from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supervised.utils import append_metrics_row, compute_metrics


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    probs: list[float] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs.extend(batch_probs.tolist())
        labels.extend(int(target) for target in targets)

    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def train_custom_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    ckpt_dir: Path,
    log_csv: Path | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_path = ckpt_dir / "custom_cnn_best.pt"
    best_auc = -math.inf

    print("Starting supervised custom CNN training...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        y_true, y_prob = evaluate_classifier(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_prob)
        scheduler.step(val_metrics["roc_auc"])

        epoch_seconds = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"CNN Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f} - "
            f"val_acc: {val_metrics['accuracy']:.6f} - val_auc: {val_metrics['roc_auc']:.6f} - "
            f"lr: {current_lr:.6f} - time: {epoch_seconds:.1f}s"
        )

        if log_csv is not None:
            append_metrics_row(
                log_csv,
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_auc": float(val_metrics["roc_auc"]),
                    "learning_rate": float(current_lr),
                    "seconds": float(epoch_seconds),
                },
            )

        if not math.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_auc:
            best_auc = float(val_metrics["roc_auc"])
            torch.save(model.state_dict(), best_path)
            print(f"New best supervised CNN AUC: {best_auc:.6f} | Saved: {best_path}")

    return best_path
