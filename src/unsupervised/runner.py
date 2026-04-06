# Histopathologic Cancer Detection - Pipeline Runner
#

# region Imports
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import roc_curve

from .datasets import ImageIdDataset, LabeledImageDataset
from .models import Autoencoder, BaselineCNN, build_resnet50_feature_extractor
from .train import eval_baseline_cnn, train_autoencoder, train_baseline_cnn, train_feature_classifier
from .utils import (
    append_metrics_row,
    compute_metrics,
    extract_features_torch,
    get_or_create_splits,
    load_features_npz,
    make_loader,
    save_features_npz,
    seed_everything,
)


# PathsAndConstants

ROOT_DIR = Path(__file__).resolve().parents[2]  
TRAIN_DIR = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"

RESULTS_DIR = ROOT_DIR / "results" / "pipeline"
SPLITS_PATH = RESULTS_DIR / "splits.npz"
METRICS_PATH = RESULTS_DIR / "metrics.csv"

AE_CKPT_DIR = RESULTS_DIR / "autoencoder" / "checkpoints"
AE_FEAT_DIR = RESULTS_DIR / "autoencoder" / "features"
TL_FEAT_DIR = RESULTS_DIR / "transfer_learning" / "features"
CNN_CKPT_DIR = RESULTS_DIR / "baseline_cnn" / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Config


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42
    overwrite_metrics_csv: bool = False

   # Train/val/test split parameters

    reuse_splits: bool = True
    max_train_samples: int = 60000
    max_val_samples: int = 15000
    max_test_samples: int = 15000

    # 1) Autoencoder (unsupervised)
    run_autoencoder: bool = True
    ae_epochs: int = 10
    ae_latent_dim: int = 256
    ae_lr: float = 1e-3
    ae_save_checkpoints: bool = True
    ae_resume: bool = True
    ae_force_reextract: bool = False

    # 2) Transfer learning (ResNet50 frozen feature extraction)
    run_transfer: bool = True
    tl_force_reextract: bool = False

    # 2b) PCA compression
    run_resnet_pca: bool = True
    resnet_pca_dim: int = 256
    resnet_pca_force: bool = False
    resnet_delete_original_after_pca: bool = False

    # 3) Baseline CNN
    run_baseline_cnn: bool = True
    cnn_epochs: int = 5
    cnn_lr: float = 1e-3

    # 4) Feature classifiers
    run_feature_classifiers: bool = True


# Main

def main() -> None:
    # Load all toggles/hyperparameters in one place (epochs, sample caps, etc.).
    cfg = PipelineConfig()
    # Make results reproducible (splits, subsampling, and model initialization).
    seed_everything(cfg.seed)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {TRAIN_DIR.resolve()}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV.resolve()}")

    # Create output folders up-front to save (npz/pt/csv/plots)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    AE_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    TL_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    CNN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if cfg.overwrite_metrics_csv and METRICS_PATH.exists():
        METRICS_PATH.unlink()

    # Prefer GPU when available; enable AMP on CUDA to speed up training/inference.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        print(f"Device: {device} | AMP: {use_amp} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: {device} | AMP: {use_amp}")

    # Load labels and create (or reuse) a fixed train/val/test split
    df = pd.read_csv(LABELS_CSV)
    splits = get_or_create_splits(
        df=df,
        seed=cfg.seed,
        max_train=cfg.max_train_samples,
        max_val=cfg.max_val_samples,
        max_test=cfg.max_test_samples,
        splits_path=SPLITS_PATH,
        reuse_existing=cfg.reuse_splits,
    )
    print("Split sizes:", {k: len(v[0]) for k, v in splits.items()})

    # Tag this run in the metrics CSV so repeated runs can be distinguished.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    metrics_rows: list[dict] = []
    # Store test probabilities so we can plot ROC curves for each method at the end.
    test_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def record_metrics(method: str, split: str, m: dict) -> None:
        # Collect metrics in-memory (for plots) and append to CSV (for reporting).
        row = {"run_id": run_id, "method": method, "split": split, **m}
        metrics_rows.append(row)
        append_metrics_row(METRICS_PATH, row)

    num_workers = 0
    ae_num_workers = 2

    # Larger batches improve GPU throughput
    ae_train_batch = 512 if device.type == "cuda" else 64
    feat_batch = 128 if device.type == "cuda" else 64
    cnn_batch = 128 if device.type == "cuda" else 32

    # Autoencoder reconstructs pixels, so keep inputs in [0,1] (no Normalize).
    ae_transform = T.Compose([T.Resize((96, 96)), T.ToTensor()])
    # Baseline CNN is supervised: augmentation helps generalization; normalize for stable optimization.
    cnn_train_transform = T.Compose(
        [
            T.Resize((96, 96)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    # Evaluation uses Non-random transform (no augmentation).
    cnn_eval_transform = T.Compose(
        [T.Resize((96, 96)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    # ResNet50 pretrained backbone expects 224×224 inputs and ImageNet normalization.
    tl_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    ae_log_csv = LOGS_DIR / "ae_history.csv"
    cnn_log_csv = LOGS_DIR / "cnn_history.csv"

    if cfg.run_autoencoder:
        run_autoencoder_step(
            cfg=cfg,
            splits=splits,
            device=device,
            use_amp=use_amp,
            ae_transform=ae_transform,
            ae_train_batch=ae_train_batch,
            feat_batch=feat_batch,
            num_workers=num_workers,
            ae_num_workers=ae_num_workers,
            log_csv=ae_log_csv,
        )

    if cfg.run_transfer:
        run_resnet_pca_step(
            cfg=cfg,
            splits=splits,
            device=device,
            tl_transform=tl_transform,
            feat_batch=feat_batch,
            num_workers=num_workers,
        )

    if cfg.run_baseline_cnn:
        run_baseline_cnn_step(
            cfg=cfg,
            splits=splits,
            device=device,
            cnn_train_transform=cnn_train_transform,
            cnn_eval_transform=cnn_eval_transform,
            cnn_batch=cnn_batch,
            num_workers=num_workers,
            log_csv=cnn_log_csv,
            record_metrics=record_metrics,
            test_probs=test_probs,
        )

    if cfg.run_feature_classifiers:
        run_feature_classifiers_step(cfg=cfg, splits=splits, record_metrics=record_metrics, test_probs=test_probs)

    save_plots_step(metrics_rows=metrics_rows, test_probs=test_probs)
    print(f"\nMetrics saved to: {METRICS_PATH}")


# Steps

def run_autoencoder_step(
    *,
    cfg: PipelineConfig,
    splits: dict,
    device: torch.device,
    use_amp: bool,
    ae_transform: T.Compose,
    ae_train_batch: int,
    feat_batch: int,
    num_workers: int,
    ae_num_workers: int,
    log_csv: Path,
) -> None:
    _banner("1) Autoencoder (Unsupervised) - Train + Extract Latents")

    # Cache check: if all latent .npz files exist, skip retraining/re-extraction to save time.

    ae_latent_paths = [
        AE_FEAT_DIR / "ae_latents_train.npz",
        AE_FEAT_DIR / "ae_latents_val.npz",
        AE_FEAT_DIR / "ae_latents_test.npz",
    ]
    if all(p.exists() for p in ae_latent_paths) and not cfg.ae_force_reextract:
        for p in ae_latent_paths:
            print(f"Skipping cached AE latents: {p}")
        _print_history_tail(log_csv, "Autoencoder training history", ["epoch", "loss", "seconds"])
        return

    # Train AE only on images (no labels needed for reconstruction loss).
    train_ids, _ = splits["train"]
    ae_train_ds = ImageIdDataset(TRAIN_DIR, train_ids, transform=ae_transform)
    ae_train_loader = make_loader(ae_train_ds, batch_size=ae_train_batch, num_workers=ae_num_workers, shuffle=True)

    # Train the autoencoder and keep the best checkpoint for feature extraction.
    ae_model = Autoencoder(latent_dim=cfg.ae_latent_dim).to(device)
    best_ae = train_autoencoder(
        model=ae_model,
        loader=ae_train_loader,
        device=device,
        epochs=cfg.ae_epochs,
        lr=cfg.ae_lr,
        use_amp=use_amp,
        ckpt_dir=AE_CKPT_DIR,
        resume=cfg.ae_resume,
        save_checkpoints=cfg.ae_save_checkpoints,
        log_csv=log_csv,
    )

    # Use the best AE weights (if checkpointing is enabled) before extracting latents.
    if best_ae is not None:
        ae_model.load_state_dict(torch.load(best_ae, map_location=device))
    # Freeze/keep only the encoder: it maps 96×96 latent vector z.
    ae_encoder = nn.Sequential(ae_model.encoder_cnn, ae_model.encoder_fc).to(device)

    # Extract and save z for each split; later supervised classifiers use these cached features.
    for split_name in ["train", "val", "test"]:
        ids, labels = splits[split_name]
        out_path = AE_FEAT_DIR / f"ae_latents_{split_name}.npz"
        # Per-split caching: skip feature extraction if the split file already exists.
        if out_path.exists() and not cfg.ae_force_reextract:
            print(f"Skipping cached AE latents: {out_path}")
            continue
        ds = ImageIdDataset(TRAIN_DIR, ids, transform=ae_transform)
        dl = make_loader(ds, batch_size=feat_batch, num_workers=num_workers, shuffle=False)
        print(f"Extracting AE latents for {split_name} (N={len(ids)})")
        out_ids, feats = extract_features_torch(ae_encoder, dl, device=device)
        save_features_npz(out_path, ids=out_ids, labels=np.asarray(labels, dtype=np.int64), features=feats)
        print(f"Saved AE latents: {out_path} | shape: {feats.shape}")


def run_resnet_pca_step(
    *,
    cfg: PipelineConfig,
    splits: dict,
    device: torch.device,
    tl_transform: T.Compose,
    feat_batch: int,
    num_workers: int,
) -> None:
    _banner("2) Transfer Learning - ResNet50 Feature Extraction (Frozen)")
    _print_resnet50_weights_status()

    # If PCA outputs exist, skips raw ResNet feature extraction (saves time/disk).
    pca_train = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_train.npz"
    pca_val = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_val.npz"
    pca_test = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_test.npz"
    pca_model = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}.joblib"
    pca_cached = pca_train.exists() and pca_val.exists() and pca_test.exists() and pca_model.exists()

    if cfg.run_resnet_pca and pca_cached and not cfg.tl_force_reextract and not cfg.resnet_pca_force:
        print(f"Skipping ResNet50 extraction (PCA cached): {pca_model}")
    else:
        # Frozen backbone: we only use ResNet50 as a feature extractor
        resnet = build_resnet50_feature_extractor().to(device)
        for p in resnet.parameters():
            p.requires_grad = False

        for split_name in ["train", "val", "test"]:
            ids, labels = splits[split_name]
            print(f"Transfer features for {split_name} (N={len(ids)})")
            maybe_extract_features(
                out_path=TL_FEAT_DIR / f"resnet50_{split_name}.npz",
                train_dir=TRAIN_DIR,
                ids=ids,
                labels=labels,
                transform=tl_transform,
                feature_model=resnet,
                device=device,
                batch_size=feat_batch,
                num_workers=num_workers,
                force=cfg.tl_force_reextract,
            )

    if cfg.run_resnet_pca:
        # PCA Compression 2048-d ResNet features down to `resnet_pca_dim` to reduce storage.
        _banner(f"2b) PCA Compression - ResNet50 Features ({cfg.resnet_pca_dim} dims)")
        _maybe_pca_compress_resnet_features(
            features_dir=TL_FEAT_DIR,
            n_components=cfg.resnet_pca_dim,
            force=cfg.resnet_pca_force,
            delete_original=cfg.resnet_delete_original_after_pca,
        )


def run_baseline_cnn_step(
    *,
    cfg: PipelineConfig,
    splits: dict,
    device: torch.device,
    cnn_train_transform: T.Compose,
    cnn_eval_transform: T.Compose,
    cnn_batch: int,
    num_workers: int,
    log_csv: Path,
    record_metrics: Callable[[str, str, dict], None],
    test_probs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    _banner("3) Baseline CNN - Train + Evaluate")

    best_baseline = CNN_CKPT_DIR / "baseline_best.pt"
    has_cnn_history = _has_csv_rows(log_csv)
    force_train_once = best_baseline.exists() and not has_cnn_history

    if best_baseline.exists() and has_cnn_history:
        print(f"Baseline CNN checkpoint (best): {best_baseline}")
        _print_history_tail(
            log_csv,
            "Baseline CNN training history",
            ["epoch", "train_loss", "val_acc", "val_auc", "seconds"],
            n=9999,
        )
    elif force_train_once:
        print(f"Baseline CNN checkpoint found but no CSV history yet: {best_baseline}")
        print("Retraining baseline CNN once to generate per-epoch history...")
    else:
        print("Baseline CNN checkpoint: none found (will train from scratch).")

    _print_baseline_checkpoint_info(CNN_CKPT_DIR)

    train_ids, train_labels = splits["train"]
    val_ids, val_labels = splits["val"]
    test_ids, test_labels = splits["test"]

    train_ds = LabeledImageDataset(TRAIN_DIR, train_ids, train_labels, transform=cnn_train_transform)
    val_ds = LabeledImageDataset(TRAIN_DIR, val_ids, val_labels, transform=cnn_eval_transform)
    test_ds = LabeledImageDataset(TRAIN_DIR, test_ids, test_labels, transform=cnn_eval_transform)

    train_dl = make_loader(train_ds, batch_size=cnn_batch, num_workers=num_workers, shuffle=True)
    val_dl = make_loader(val_ds, batch_size=cnn_batch, num_workers=num_workers, shuffle=False)
    test_dl = make_loader(test_ds, batch_size=cnn_batch, num_workers=num_workers, shuffle=False)

    cnn_model = BaselineCNN().to(device)
    if best_baseline.exists() and has_cnn_history and not force_train_once:
        best_cnn = best_baseline
    else:
        if log_csv.exists():
            log_csv.unlink()
        best_cnn = train_baseline_cnn(
            model=cnn_model,
            train_loader=train_dl,
            val_loader=val_dl,
            device=device,
            epochs=cfg.cnn_epochs,
            lr=cfg.cnn_lr,
            ckpt_dir=CNN_CKPT_DIR,
            resume=False,
            log_csv=log_csv,
        )

    print(f"Loading baseline CNN weights: {best_cnn}")
    cnn_model.load_state_dict(torch.load(best_cnn, map_location=device))
    y_true, y_prob = eval_baseline_cnn(cnn_model, test_dl, device=device)
    m = compute_metrics(y_true, y_prob)
    record_metrics("baseline_cnn", "test", m)
    test_probs["baseline_cnn"] = (y_true, y_prob)
    print(f"Baseline CNN test metrics: {m}")


def run_feature_classifiers_step(
    *,
    cfg: PipelineConfig,
    splits: dict,
    record_metrics: Callable[[str, str, dict], None],
    test_probs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    _banner("4) Supervised Classifiers on Cached Features - Evaluate + Compare")

    ae_train = AE_FEAT_DIR / "ae_latents_train.npz"
    ae_val = AE_FEAT_DIR / "ae_latents_val.npz"
    ae_test = AE_FEAT_DIR / "ae_latents_test.npz"
    if ae_train.exists() and ae_val.exists() and ae_test.exists():
        _, y_train, x_train = load_features_npz(ae_train)
        _, y_val, x_val = load_features_npz(ae_val)
        _, y_test, x_test = load_features_npz(ae_test)
        clf = train_feature_classifier(x_train, y_train)
        val_prob = clf.predict_proba(x_val)[:, 1].astype(np.float32)
        test_prob = clf.predict_proba(x_test)[:, 1].astype(np.float32)
        m_val = compute_metrics(y_val, val_prob)
        m_test = compute_metrics(y_test, test_prob)
        record_metrics("ae_latents", "val", m_val)
        record_metrics("ae_latents", "test", m_test)
        test_probs["ae_latents"] = (y_test, test_prob)
        print(f"ae_latents val metrics: {m_val}")
        print(f"ae_latents test metrics: {m_test}")

    # Prefer PCA-compressed features if present; otherwise fall back to raw ResNet50 features.
    rn_train = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_train.npz"
    rn_val = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_val.npz"
    rn_test = TL_FEAT_DIR / f"resnet50_pca{cfg.resnet_pca_dim}_test.npz"
    if not (rn_train.exists() and rn_val.exists() and rn_test.exists()):
        rn_train = TL_FEAT_DIR / "resnet50_train.npz"
        rn_val = TL_FEAT_DIR / "resnet50_val.npz"
        rn_test = TL_FEAT_DIR / "resnet50_test.npz"

    if rn_train.exists() and rn_val.exists() and rn_test.exists():
        _, y_train, x_train = load_features_npz(rn_train)
        _, y_val, x_val = load_features_npz(rn_val)
        _, y_test, x_test = load_features_npz(rn_test)
        clf = train_feature_classifier(x_train, y_train)
        val_prob = clf.predict_proba(x_val)[:, 1].astype(np.float32)
        test_prob = clf.predict_proba(x_test)[:, 1].astype(np.float32)
        m_val = compute_metrics(y_val, val_prob)
        m_test = compute_metrics(y_test, test_prob)
        record_metrics("resnet50_feats", "val", m_val)
        record_metrics("resnet50_feats", "test", m_test)
        test_probs["resnet50_feats"] = (y_test, test_prob)
        print(f"resnet50_feats val metrics: {m_val}")
        print(f"resnet50_feats test metrics: {m_test}")


def save_plots_step(*, metrics_rows: list[dict], test_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    _banner("Summary - Plots")
    plots_dir = RESULTS_DIR / "plots"
    _plot_test_comparison(metrics_rows, plots_dir)
    _plot_roc_curves(test_probs, plots_dir)


# Helpers


def _banner(title: str) -> None:
    line = "=" * 72
    print("\n" + line)
    print(title)
    print(line)


def _print_resnet50_weights_status() -> None:
    try:
        checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
        matches = sorted(checkpoints_dir.glob("resnet50-*.pth"))
        if matches:
            print(f"ResNet50 pretrained weights (cached): {matches[-1]}")
        else:
            print("ResNet50 pretrained weights not found in cache (will download on first use).")
    except Exception:
        print("ResNet50 pretrained weights: status unknown (could not check cache).")


def _print_history_tail(path: Path, title: str, cols: list[str], n: int = 5) -> None:
    if not path.exists():
        return
    try:
        df = pd.read_csv(path)
        if df.empty:
            return
        tail = df.tail(n)
        keep_cols = [c for c in cols if c in tail.columns]
        if not keep_cols:
            return
        print(f"{title} (last {len(tail)} rows):")
        print(tail[keep_cols].to_string(index=False))
    except Exception:
        return


def _has_csv_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except Exception:
        return False


def _print_baseline_checkpoint_info(ckpt_dir: Path) -> None:
    best_path = ckpt_dir / "baseline_best.pt"
    if best_path.exists():
        print(f"Baseline CNN best-weights file: {best_path}")
    else:
        print("Baseline CNN best-weights file: none found (will train).")


def _plot_test_comparison(metrics_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    test_rows = [r for r in metrics_rows if r.get("split") == "test"]
    if not test_rows:
        return

    df = pd.DataFrame(test_rows)
    df = df.sort_values("roc_auc", ascending=False)

    methods = df["method"].tolist()
    roc_auc = df["roc_auc"].astype(float).to_numpy()
    f1 = df["f1"].astype(float).to_numpy()

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, roc_auc, width, label="ROC-AUC")
    ax.bar(x + width / 2, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test Set Comparison (higher is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "comparison_test.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _plot_roc_curves(test_probs: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not test_probs:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for method, (y_true, y_prob) in test_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = compute_metrics(y_true, y_prob)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{method} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "roc_curves_test.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _maybe_pca_compress_resnet_features(
    features_dir: Path,
    n_components: int,
    force: bool,
    delete_original: bool,
) -> tuple[Path, Path, Path] | None:
    out_train = features_dir / f"resnet50_pca{n_components}_train.npz"
    out_val = features_dir / f"resnet50_pca{n_components}_val.npz"
    out_test = features_dir / f"resnet50_pca{n_components}_test.npz"
    pca_path = features_dir / f"resnet50_pca{n_components}.joblib"

    if out_train.exists() and out_val.exists() and out_test.exists() and pca_path.exists() and not force:
        print(f"Skipping PCA (cached): {pca_path}")
        return out_train, out_val, out_test

    in_train = features_dir / "resnet50_train.npz"
    in_val = features_dir / "resnet50_val.npz"
    in_test = features_dir / "resnet50_test.npz"
    if not (in_train.exists() and in_val.exists() and in_test.exists()):
        print("PCA needs resnet50_{train,val,test}.npz files, but they were not found.")
        return None

    print(f"Fitting PCA on ResNet50 train features -> {n_components} dims")
    _, _, x_tr = load_features_npz(in_train)
    x_tr = x_tr.astype(np.float32, copy=False)

    ipca = IncrementalPCA(n_components=n_components, batch_size=2048)
    for i in range(0, x_tr.shape[0], ipca.batch_size):
        ipca.partial_fit(x_tr[i : i + ipca.batch_size])

    joblib.dump(ipca, pca_path)
    print(f"Saved PCA model: {pca_path}")

    def transform_and_save(in_path: Path, out_path: Path) -> None:
        ids, y, x = load_features_npz(in_path)
        x = x.astype(np.float32, copy=False)
        z = ipca.transform(x).astype(np.float32, copy=False)
        save_features_npz(out_path, ids=ids, labels=y, features=z)
        print(f"Saved PCA features: {out_path} | shape: {z.shape}")

    transform_and_save(in_train, out_train)
    transform_and_save(in_val, out_val)
    transform_and_save(in_test, out_test)

    if delete_original:
        in_train.unlink(missing_ok=True)
        in_val.unlink(missing_ok=True)
        in_test.unlink(missing_ok=True)
        print("Deleted original ResNet50 feature files to save disk space.")

    return out_train, out_val, out_test


def maybe_extract_features(
    out_path: Path,
    train_dir: Path,
    ids: list[str],
    labels: np.ndarray,
    transform: T.Compose,
    feature_model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        print(f"Skipping cached features: {out_path}")
        return
    ds = ImageIdDataset(train_dir, ids, transform=transform)
    dl = make_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    out_ids, feats = extract_features_torch(feature_model, dl, device=device)
    save_features_npz(out_path, ids=out_ids, labels=np.asarray(labels, dtype=np.int64), features=feats)
    print(f"Saved features: {out_path} | shape: {feats.shape}")


# Main

if __name__ == "__main__":
    main()
