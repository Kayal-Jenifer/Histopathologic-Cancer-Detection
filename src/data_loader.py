# Dataset: Histopathologic Cancer Detection (Kaggle)
# Task: Binary image classification — cancer (1) or no cancer (0)

import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# Root of the project (one level above this src/ file)
ROOT_DIR   = Path(__file__).resolve().parent.parent

# Paths to the dataset — update if your folder is different
TRAIN_DIR  = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"
PLOTS_DIR = ROOT_DIR / "results" / "screenshots"

LABEL_NEG = "No Cancer"
LABEL_POS = "Cancer"

# Normalisation values computed specifically for this dataset
# Using these instead of ImageNet values gives better results for histopathology images
MEAN = (0.7008, 0.5384, 0.6916)  # mean pixel value per channel (R, G, B)
STD  = (0.2350, 0.2774, 0.2128)  # standard deviation per channel (R, G, B)

# Training transforms include augmentations to help the model generalise
# Flips and rotations are safe because tissue has no fixed orientation
# ColorJitter handles stain colour variation across different lab slides
train_transforms = T.Compose([
    T.Resize((96, 96)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),                   # converts PIL image to tensor [0.0 - 1.0]
    T.Normalize(MEAN, STD),         # standardises pixel values
])

# Validation and test transforms — no augmentation, just resize and normalise
val_transforms = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


# Custom Dataset class
# Images are loaded one at a time (not all at once) because 220k images won't fit in memory
class CancerDataset(Dataset):
    def __init__(self, ids, labels, transform):
        self.ids       = ids
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load image from disk, convert to RGB, apply transforms
        image = Image.open(TRAIN_DIR / f"{self.ids[idx]}.tif").convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


if __name__ == "__main__":
    # Fix random seeds so results are the same every run
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate image directory exists before doing anything else
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {TRAIN_DIR.resolve()}\n"
            "Download the dataset from Kaggle and place the .tif files in data/train/"
        )

    # ── 1. Dataset Overview ──────────────────────────────────────────────────
    print("=" * 50)
    print("  1. DATASET OVERVIEW")
    print("=" * 50)
    df = pd.read_csv(LABELS_CSV)
    print(f"  Total images : {len(df):,}")
    print(df["label"].value_counts().to_string())

    # ── 2. Train (75%) / Val (15%) / Test Split (10%) ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  2. TRAIN (75%) / VAL (15%) / TEST SPLIT (10%)")
    print("=" * 50)
    train_val, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=SEED)
    train, val      = train_test_split(train_val, test_size=0.1667, stratify=train_val["label"], random_state=SEED)
    print(f"  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    print(f"  Test  : {len(test):,}")

    # ── 3. DataLoader Configuration ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  3. DATALOADER CONFIGURATION")
    print("=" * 50)
    NUM_WORKERS = min(multiprocessing.cpu_count(), 8)
    PIN_MEMORY  = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": 32,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(CancerDataset(train["id"].tolist(), train["label"].tolist(), train_transforms), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(CancerDataset(val["id"].tolist(),   val["label"].tolist(),   val_transforms),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(CancerDataset(test["id"].tolist(),  test["label"].tolist(),  val_transforms),   shuffle=False, **loader_kwargs)

    print(f"  Workers     : {NUM_WORKERS}")
    print(f"  pin_memory  : {PIN_MEMORY}")
    print("  Batch size  : 32")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # ── 4. Batch Sanity Check ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  4. BATCH SANITY CHECK")
    print("=" * 50)
    images, labels = next(iter(train_loader))
    print(f"  Image shape : {tuple(images.shape)}")
    print(f"  Label shape : {tuple(labels.shape)}")
    print(f"  Pixel range : [{images.min():.2f}, {images.max():.2f}]")

    # ── 5. Class Balance ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  5. CLASS BALANCE")
    print("=" * 50)
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    print(f"  {LABEL_NEG} : {counts[0]:,}  ({counts[0]/total*100:.1f}%)")
    print(f"  {LABEL_POS}    : {counts[1]:,}  ({counts[1]/total*100:.1f}%)")
    print(f"  Imbalance ratio : {counts[0]/counts[1]:.2f}:1")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([LABEL_NEG, LABEL_POS], counts.values, color=["#4C9BE8", "#E8714C"])
    axes[0].set_title("Class Distribution (counts)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=10)

    axes[1].pie(counts.values, labels=[LABEL_NEG, LABEL_POS],
                colors=["#4C9BE8", "#E8714C"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class Distribution (%)")
    plt.suptitle("Class Balance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=150)
    plt.show()
    plt.close()

    # ── Plot 2: Train / Val / Test Split ─────────────────────────────────────
    # Confirms the class ratio is preserved across all three splits
    fig, ax = plt.subplots(figsize=(7, 4))
    splits = ["Train", "Val", "Test"]
    neg_c  = [train["label"].value_counts()[0], val["label"].value_counts()[0], test["label"].value_counts()[0]]
    pos_c  = [train["label"].value_counts()[1], val["label"].value_counts()[1], test["label"].value_counts()[1]]
    x = np.arange(3)
    ax.bar(x - 0.2, neg_c, 0.4, label=LABEL_NEG, color="#4C9BE8")
    ax.bar(x + 0.2, pos_c, 0.4, label=LABEL_POS, color="#E8714C")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Train / Val / Test Split by Class")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "split_distribution.png", dpi=150)
    plt.show()
    plt.close()

    # ── Plot 3: Sample Images ────────────────────────────────────────────────
    # Visual proof that images loaded correctly — 4 no-cancer and 4 cancer patches
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    fig.suptitle(f"Sample Patches — {LABEL_NEG} (top) vs {LABEL_POS} (bottom)", fontsize=12, fontweight="bold")

    for col, img_id in enumerate(df[df["label"] == 0]["id"].sample(4, random_state=SEED)):
        img = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{LABEL_NEG}\n{img_id[:8]}…", fontsize=8)
        axes[0, col].axis("off")

    for col, img_id in enumerate(df[df["label"] == 1]["id"].sample(4, random_state=SEED)):
        img = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"{LABEL_POS}\n{img_id[:8]}…", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sample_images.png", dpi=150)
    plt.show()
    plt.close()
    print("Pipeline complete.")
