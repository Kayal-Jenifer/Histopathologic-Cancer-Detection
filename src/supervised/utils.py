import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def append_metrics_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


def subsample_stratified(df_split: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(df_split) <= n:
        return df_split
    subset, _ = train_test_split(
        df_split,
        train_size=n,
        stratify=df_split["label"],
        random_state=seed,
    )
    return subset


def save_splits(path: Path, splits: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_ids=np.asarray(splits["train"][0]),
        train_labels=np.asarray(splits["train"][1], dtype=np.int64),
        val_ids=np.asarray(splits["val"][0]),
        val_labels=np.asarray(splits["val"][1], dtype=np.int64),
        test_ids=np.asarray(splits["test"][0]),
        test_labels=np.asarray(splits["test"][1], dtype=np.int64),
    )


def load_splits(path: Path) -> dict:
    blob = np.load(path, allow_pickle=False)
    return {
        "train": (blob["train_ids"].tolist(), blob["train_labels"]),
        "val": (blob["val_ids"].tolist(), blob["val_labels"]),
        "test": (blob["test_ids"].tolist(), blob["test_labels"]),
    }


def get_or_create_splits(
    df: pd.DataFrame,
    seed: int,
    max_train,
    max_val,
    max_test,
    splits_path: Path,
    reuse_existing: bool,
) -> dict:
    if reuse_existing and splits_path.exists():
        print(f"Loading cached splits: {splits_path}")
        return load_splits(splits_path)

    train_val, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=0.1667, stratify=train_val["label"], random_state=seed)

    train = subsample_stratified(train, max_train, seed=seed)
    val = subsample_stratified(val, max_val, seed=seed)
    test = subsample_stratified(test, max_test, seed=seed)

    splits = {
        "train": (train["id"].tolist(), train["label"].to_numpy(dtype=np.int64)),
        "val": (val["id"].tolist(), val["label"].to_numpy(dtype=np.int64)),
        "test": (test["id"].tolist(), test["label"].to_numpy(dtype=np.int64)),
    }
    save_splits(splits_path, splits)
    print(f"Saved splits: {splits_path}")
    return splits
