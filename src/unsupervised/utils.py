import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import torch
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


def subsample_stratified(df_split: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
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
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
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


def latest_checkpoint(ckpt_dir: Path, prefix: str) -> Path | None:
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob(f"{prefix}_epoch*.pt"))
    if not candidates:
        return None

    def epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("epoch")[-1])
        except ValueError:
            return -1

    candidates = sorted(candidates, key=epoch_num)
    return candidates[-1] if epoch_num(candidates[-1]) >= 0 else None


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

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": auc,
    }


def save_features_npz(path: Path, ids: np.ndarray, labels: np.ndarray, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, labels=np.asarray(labels, dtype=np.int64), features=features)


def load_features_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path, allow_pickle=False)
    return blob["ids"], blob["labels"], blob["features"]


@torch.inference_mode()
def extract_features_torch(
    feature_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_model.eval()
    all_ids: list[str] = []
    feats: list[torch.Tensor] = []

    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        f = feature_model(x).detach().cpu()
        feats.append(f)
        all_ids.extend(list(batch_ids))

    features = torch.cat(feats, dim=0).numpy().astype(np.float32)
    ids_arr = np.asarray(all_ids)
    return ids_arr, features
