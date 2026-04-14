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


# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"
RESULTS_DIR = ROOT_DIR / "results" / "supervised_learning"
TABLES_DIR = RESULTS_DIR / "tables"
SCREENSHOTS_DIR = ROOT_DIR / "results" / "screenshots"


# Labels and normalization
LABEL_NEG = "No Cancer"
LABEL_POS = "Cancer"
MEAN = (0.7008, 0.5384, 0.6916)
STD = (0.2350, 0.2774, 0.2128)


# Base settings
SEED = 42
SAMPLES_PER_CLASS = 1_000
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
NUM_WORKERS = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_transforms():
    train_transforms = T.Compose(
        [
            T.Resize((96, 96)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    eval_transforms = T.Compose([T.Resize((96, 96)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_transforms, eval_transforms


def make_splits():
    df_full = pd.read_csv(LABELS_CSV)
    neg_sample = df_full[df_full["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=SEED)
    pos_sample = df_full[df_full["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=SEED)
    df = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_val, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.1667, stratify=train_val["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


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


def train_one_experiment(config, train_loader, val_loader, device):
    model = CustomCNN(
        filters=config["filters"],
        dense_units=config["dense_units"],
        dropout=config["dropout"],
        use_batchnorm=config["use_batchnorm"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
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

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

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

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history_rows


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
        plt.close()


def main():
    seed_everything(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

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

    experiments = [
        {
            "name": "baseline",
            "filters": (32, 64, 128, 256),
            "dense_units": 256,
            "dropout": 0.40,
            "use_batchnorm": True,
            "learning_rate": 1e-3,
            "epochs": EPOCHS,
        },
        {
            "name": "fewer_layers",
            "filters": (32, 64, 128),
            "dense_units": 256,
            "dropout": 0.30,
            "use_batchnorm": True,
            "learning_rate": 1e-3,
            "epochs": EPOCHS,
        },
        {
            "name": "no_batchnorm",
            "filters": (32, 64, 128, 256),
            "dense_units": 256,
            "dropout": 0.40,
            "use_batchnorm": False,
            "learning_rate": 1e-3,
            "epochs": EPOCHS,
        },
        {
            "name": "lower_lr_deeper",
            "filters": (32, 64, 128, 256, 512),
            "dense_units": 256,
            "dropout": 0.50,
            "use_batchnorm": True,
            "learning_rate": 5e-4,
            "epochs": EPOCHS,
        },
    ]

    history_rows = []
    summary_rows = []

    for config in experiments:
        best_model, experiment_history = train_one_experiment(config, train_loader, val_loader, device)
        history_rows.extend(experiment_history)

        test_metrics = evaluate_model(best_model, test_loader, device)
        summary_rows.append(
            {
                "experiment": config["name"],
                "filters": str(config["filters"]),
                "dense_units": config["dense_units"],
                "dropout": config["dropout"],
                "use_batchnorm": config["use_batchnorm"],
                "learning_rate": config["learning_rate"],
                "epochs": config["epochs"],
                **test_metrics,
            }
        )
        print(f"  Test metrics    : {test_metrics}")

    pd.DataFrame(history_rows).to_csv(TABLES_DIR / "supervised_history.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "supervised_results.csv", index=False)
    save_training_plots(history_rows)

    print("\n" + "=" * 55)
    print("  SUPERVISED LEARNING COMPLETE")
    print("=" * 55)
    print(f"  History saved : {TABLES_DIR / 'supervised_history.csv'}")
    print(f"  Results saved : {TABLES_DIR / 'supervised_results.csv'}")
    print(f"  Plots saved   : {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
