# Supervised Learning - custom CNN pipeline for binary image classification

import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"
RESULTS_DIR = ROOT_DIR / "results" / "supervised_learning"
SCREENSHOTS_DIR = ROOT_DIR / "results" / "screenshots"


# ─────────────────────────────────────────────────────────────────────────────
#  Labels and normalization
# ─────────────────────────────────────────────────────────────────────────────
# These normalization values are reused from the project data-preparation work.
LABEL_NEG = "No Cancer"
LABEL_POS = "Cancer"
MEAN = (0.7008, 0.5384, 0.6916)
STD = (0.2350, 0.2774, 0.2128)


# ─────────────────────────────────────────────────────────────────────────────
#  Base settings
# ─────────────────────────────────────────────────────────────────────────────
# We keep the supervised task aligned with data exploration by using
# a balanced subset of 1,000 images per class = 2,000 total images.
SEED = 42
SAMPLES_PER_CLASS = 1_000
BATCH_SIZE = 32 #64
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_WORKERS = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility helper
# ─────────────────────────────────────────────────────────────────────────────
# Fix random seeds so the same subset and splits are produced every run.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Custom Dataset
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch loads images one at a time instead of keeping everything in memory.
class CancerDataset(Dataset):
    def __init__(self, ids, labels, transform):
        self.ids = list(ids)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = Image.open(TRAIN_DIR / f"{self.ids[idx]}.tif").convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
#  CNN Building Blocks
# ─────────────────────────────────────────────────────────────────────────────
# Each convolution block learns local tissue patterns and then reduces spatial size with max pooling.
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(dropout)])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# The full custom CNN stacks several ConvBlocks, then uses a dense layer
# to convert learned image features into one cancer / no-cancer prediction.
class CustomCNN(nn.Module):
    def __init__(self, filters=(32, 64, 128, 256), dense_units=256, dropout=0.4, use_batchnorm=True):
        super().__init__()
        blocks = []
        in_channels = 3

        for out_channels in filters:
            block_dropout = dropout * 0.5 if out_channels < 128 else dropout
            blocks.append(ConvBlock(in_channels, out_channels, block_dropout, use_batchnorm))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Image transforms
# ─────────────────────────────────────────────────────────────────────────────
# Training uses augmentation; validation/test use only resize + normalization.
def make_transforms():
    train_transforms = T.Compose(
        [
            T.Resize((96, 96)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    eval_transforms = T.Compose([T.Resize((96, 96)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_transforms, eval_transforms


# ─────────────────────────────────────────────────────────────────────────────
#  Balanced 2,000-image subset + train/val/test split
# ─────────────────────────────────────────────────────────────────────────────
# Similar idea used in data_exploration 1,000 no-cancer + 1,000 cancer images, then a stratified split.
def make_splits():
    df_full = pd.read_csv(LABELS_CSV)
    existing = {p.stem for p in TRAIN_DIR.glob("*.tif")}
    df_full = df_full[df_full["id"].isin(existing)].reset_index(drop=True)
    neg_sample = df_full[df_full["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=SEED)
    pos_sample = df_full[df_full["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=SEED)
    df = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.1667, stratify=train_val["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
# Wrap the datasets into batched PyTorch loaders for training/evaluation.
def make_loaders(batch_size: int):
    train_df, val_df, test_df = make_splits()
    train_tfms, eval_tfms = make_transforms()

    train_dataset = CancerDataset(train_df["id"], train_df["label"], train_tfms)
    val_dataset = CancerDataset(val_df["id"], val_df["label"], eval_tfms)
    test_dataset = CancerDataset(test_df["id"], test_df["label"], eval_tfms)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_df, val_df, test_df, train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
# Run the model on validation or test data and compute classification metrics.
@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  One experiment = train one CNN configuration
# ─────────────────────────────────────────────────────────────────────────────
# This function trains one model setting (for example baseline, fewer layers,
# or no batch normalization), keeps the best epoch, and returns its history.
def train_one_experiment(config, train_loader, val_loader, device):
    model = CustomCNN(
        filters=config["filters"],
        dense_units=config["dense_units"],
        dropout=config["dropout"],
        use_batchnorm=config["use_batchnorm"],
    ).to(device)
    
    #removed previous code: optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # to expand with the folllowng optimizations comparing Adam with SGD 
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"]
        )

    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9
        )

    best_state = None
    best_val_auc = -float("inf")
    history_rows = []

    print("\n" + "=" * 55)
    print(f"  EXPERIMENT: {config['name']}")
    print("=" * 55)
    print(f"  Filters       : {config['filters']}")
    print(f"  Dense units   : {config['dense_units']}")
    print(f"  Dropout       : {config['dropout']}")
    print(f"  BatchNorm     : {config['use_batchnorm']}")
    print(f"  Learning rate : {config['learning_rate']}")

    # Standard training loop: train on all batches, then validate once per epoch.
    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        # Forward pass -> loss -> backpropagation -> optimizer step.
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        # After each epoch, measure how well the current model generalizes.
        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        elapsed = time.time() - start_time

        print(
            f"  Epoch {epoch}/{config['epochs']} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_acc: {val_metrics['accuracy']:.4f} - "
            f"val_auc: {val_metrics['roc_auc']:.4f} - "
            f"time: {elapsed:.1f}s"
        )

        history_rows.append(
            {
                "experiment": config["name"],
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["roc_auc"],
                "seconds": elapsed,
            }
        )

        # Keep the best model weights based on validation ROC-AUC.
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history_rows


# ─────────────────────────────────────────────────────────────────────────────
#  Save training curves
# ─────────────────────────────────────────────────────────────────────────────
# Each experiment gets a loss plot and a validation-metrics plot.
def save_training_plots(history_rows):
    history_df = pd.DataFrame(history_rows)

    for experiment_name in history_df["experiment"].unique():
        run_df = history_df[history_df["experiment"] == experiment_name].copy()

        plt.figure(figsize=(8, 5))
        plt.plot(run_df["epoch"], run_df["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{experiment_name} - Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(SCREENSHOTS_DIR / f"{experiment_name}_training_loss.png", dpi=150)
        plt.show()
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(run_df["epoch"], run_df["val_accuracy"], marker="o", label="Validation Accuracy")
        plt.plot(run_df["epoch"], run_df["val_auc"], marker="o", label="Validation ROC-AUC")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{experiment_name} - Validation Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(SCREENSHOTS_DIR / f"{experiment_name}_validation_metrics.png", dpi=150)
        plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Supervised-learning pipeline
# ─────────────────────────────────────────────────────────────────────────────
# 1. Check dataset paths
# 2. Build the balanced 2K subset and split it
# 3. Train several CNN variants
# 4. Compare their results and save plots
def main():
    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fail early if the expected dataset files are missing.
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {TRAIN_DIR.resolve()}\n"
            "Place the extracted .tif files in data/train/"
        )
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("  SUPERVISED LEARNING - CUSTOM CNN")
    print("=" * 55)
    print(f"  Device : {device}")
    print(f"  Balanced subset : {SAMPLES_PER_CLASS:,} per class ({SAMPLES_PER_CLASS*2:,} total)")

    train_df, val_df, test_df, train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE)
    print(f"  Train samples : {len(train_df):,}")
    print(f"  Val samples   : {len(val_df):,}")
    print(f"  Test samples  : {len(test_df):,}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Workers       : {NUM_WORKERS}")

    # These experiment settings are here because the task explicitly asks us
    # to compare different architecture and training choices.
    experiments = [

        {
        "name": "baseline_adam",
        "filters": (32,64,128,256),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        },

        {
        "name": "adam_lower_lr",
        "filters": (32,64,128,256),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.0005,
        "optimizer": "Adam",
        "epochs": EPOCHS
        },

        {
        "name": "sgd_model",
        "filters": (32,64,128,256),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.01,
        "optimizer": "SGD",
        "epochs": EPOCHS
        },

        {
        "name": "larger_dense",
        "filters": (32,64,128,256),
        "dense_units": 512,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        },

        {
        "name": "higher_dropout",
        "filters": (32,64,128,256),
        "dense_units": 256,
        "dropout": 0.50,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        }
        ,

        {
        "name": "fewer_layers",
        "filters": (32,64,128),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        },

        {
        "name": "deeper_model",
        "filters": (32,64,128,256,512),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        },

        {
        "name": "wider_filters",
        "filters": (64,128,256,512),
        "dense_units": 256,
        "dropout": 0.40,
        "use_batchnorm": True,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "epochs": EPOCHS
        }

        ]

    history_rows = []
    results_rows = []   
    # Train and evaluate each experiment one after another.
    for config in experiments:
        best_model, experiment_history = train_one_experiment(config, train_loader, val_loader, device)
        history_rows.extend(experiment_history)

        test_metrics = evaluate_model(best_model, test_loader, device)
        print(f"  Test metrics    : {test_metrics}")

        results_rows.append({
            "Model": config["name"],
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1": test_metrics["f1"],
            "ROC_AUC": test_metrics["roc_auc"],
        })

# Save screenshots for the per-experiment training curves.
    save_training_plots(history_rows)

    save_comparison_plots(results_rows, history_rows)
    #─────────────────────────────────────────────────────────────────────────────
    #  Comparison plots across all experiments (hyperparameter comparison visuals)
    # ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plots(results_rows, history_rows):
    results = pd.DataFrame(results_rows)
    history_df = pd.DataFrame(history_rows)

    results.to_csv(RESULTS_DIR / "cnn_hyperparameter_comparison.csv", index=False)

    # ── Chart 1: All 5 metrics grouped bar chart ─────────────────────────────
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(results))
    width = 0.15
    colors = ["#2C3E50", "#566573", "#808B96", "#AAB7B8", "#D5D8DC"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, results[metric], width, label=metric, color=colors[i], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=20, ha="right")
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Supervised CNN — Hyperparameter Comparison (All Test Metrics)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 2: Accuracy comparison ──────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    bars = plt.bar(results["Model"], results["Accuracy"], color="#2C3E50", edgecolor="white", width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    plt.title("Model Accuracy Comparison", fontweight="bold")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 3: ROC-AUC comparison ───────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    bars = plt.bar(results["Model"], results["ROC_AUC"], color="#566573", edgecolor="white", width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.0005,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=10)
    plt.title("Model ROC-AUC Comparison", fontweight="bold")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_roc_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 4: Recall comparison (most important) ────────────────
    plt.figure(figsize=(10, 5))
    max_recall = results["Recall"].max()
    bar_colors = ["#2C3E50" if r == max_recall else "#808B96" for r in results["Recall"]]
    bars = plt.bar(results["Model"], results["Recall"], color=bar_colors, edgecolor="white", width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    plt.axhline(y=max_recall, color="#2C3E50", linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Best recall = {max_recall:.3f}")
    plt.title("Cancer Detection Recall Comparison\n(Higher = fewer missed cancers)", fontweight="bold")
    plt.ylabel("Recall (Sensitivity)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_recall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 5: Combined training loss curves across all experiments ────────
    plt.figure(figsize=(11, 6))
    for experiment_name in history_df["experiment"].unique():
        run_df = history_df[history_df["experiment"] == experiment_name]
        plt.plot(run_df["epoch"], run_df["train_loss"], marker="o", label=experiment_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss — All Hyperparameter Configurations", fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 6: Combined validation AUC curves ──────────────────────────────
    plt.figure(figsize=(11, 6))
    for experiment_name in history_df["experiment"].unique():
        run_df = history_df[history_df["experiment"] == experiment_name]
        plt.plot(run_df["epoch"], run_df["val_auc"], marker="o", label=experiment_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC-AUC")
    plt.title("Validation ROC-AUC — All Hyperparameter Configurations", fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_val_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 7: Precision vs. Recall scatter (trade-off) ───────────
    plt.figure(figsize=(9, 7))
    for _, row in results.iterrows():
        plt.scatter(row["Recall"], row["Precision"], s=160, edgecolor="black", zorder=3)
        plt.annotate(row["Model"], (row["Recall"], row["Precision"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=9)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision vs. Recall Trade-off Across Experiments", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 8: Heatmap summary of all metrics by experiment ────────────────
    heatmap_data = results[metrics].values
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto", vmin=0.60, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels(results["Model"])
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            color = "white" if val > 0.82 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="Metric Score")
    ax.set_title("Hyperparameter Results Heatmap — All Metrics × All Experiments", fontweight="bold")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "cnn_comparison_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Comparison plots saved to {SCREENSHOTS_DIR}")
    print(f"  Comparison CSV saved to {RESULTS_DIR / 'cnn_hyperparameter_comparison.csv'}")

    print("\n" + "=" * 55)
    print("  SUPERVISED LEARNING COMPLETE")
    print("=" * 55)
    print(f"  Plots saved   : {SCREENSHOTS_DIR}")

if __name__ == "__main__":
    main()