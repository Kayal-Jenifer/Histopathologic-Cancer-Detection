# Data Exploration — Histopathologic Cancer Detection
# Subset: 1 000 cancer + 1 000 no-cancer = 2 000 images total

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as T  # used only in section 9 (augmentation preview)

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve paths relative to this file so the script works from any directory
ROOT_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"
PLOTS_DIR  = ROOT_DIR / "results" / "screenshots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable class names used in plot titles and console output
LABEL_NEG = "No Cancer"
LABEL_POS = "Cancer"

SEED = 42              # fixed seed for reproducible sampling and shuffling
SAMPLES_PER_CLASS = 1_000  # 1 000 per class → 2 000 total subset

# Seed all libraries that have their own internal RNG state
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: load a list of image IDs as raw numpy arrays (H×W×C, uint8)
# ─────────────────────────────────────────────────────────────────────────────
def load_images(ids):
    # Keep as uint8 (0–255) to avoid doubling memory with a float32 cast upfront.
    # Callers that need float should divide by 255 themselves.
    return [np.array(Image.open(TRAIN_DIR / f"{i}.tif").convert("RGB")) for i in ids]


# ─────────────────────────────────────────────────────────────────────────────
#  1. Build the 2 000-image subset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  HISTOPATHOLOGIC CANCER DETECTION — DATA EXPLORATION")
print(f"  Subset: {SAMPLES_PER_CLASS:,} per class  ({SAMPLES_PER_CLASS*2:,} total)")
print("=" * 55)

# Fail early with a helpful message rather than a cryptic FileNotFoundError later
if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found: {TRAIN_DIR.resolve()}\n"
        "Download the dataset from Kaggle and place the .tif files in data/train/"
    )

df_full = pd.read_csv(LABELS_CSV)
print(f"\n  Full dataset : {len(df_full):,} images")
print(f"  Label counts :\n{df_full['label'].value_counts().to_string()}")

# Only keep rows whose .tif file is actually present on disk
available_ids = {p.stem for p in TRAIN_DIR.glob("*.tif")}
df_full = df_full[df_full["id"].isin(available_ids)]
print(f"  Available    : {len(df_full):,} images (files present on disk)")

# Sample equal numbers from each class to create a perfectly balanced subset.
# random_state=SEED ensures the same images are picked on every run.
neg_sample = df_full[df_full["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=SEED)
pos_sample = df_full[df_full["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=SEED)

# Concatenate then shuffle so rows alternate randomly rather than all 0s then all 1s
df = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"\n  Subset       : {len(df):,} images  ({SAMPLES_PER_CLASS:,} per class)")

# Separate IDs by class — used throughout for per-class operations
neg_ids = df[df["label"] == 0]["id"].tolist()
pos_ids = df[df["label"] == 1]["id"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  2. Image Shape & Size Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. IMAGE SHAPE & SIZE SANITY CHECK")
print("=" * 55)

# Check 20 randomly selected images — enough to catch any inconsistent shapes
# without paying the full cost of loading all 2 000 images twice
sample_check_ids = df["id"].sample(20, random_state=SEED).tolist()
shapes = set()
for img_id in sample_check_ids:
    arr = np.array(Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB"))
    shapes.add(arr.shape)

print(f"  Unique shapes found (20-image sample): {shapes}")
print(f"  Expected : (96, 96, 3)")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Sample Images — raw patches
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Loading sample images for visualisation...")

# Show the first 8 IDs from each class (deterministic because neg/pos_ids
# were built from a seeded sample, so these are always the same images)
NEG_DISPLAY = neg_ids[:8]
POS_DISPLAY = pos_ids[:8]

fig, axes = plt.subplots(2, 8, figsize=(18, 5))
fig.suptitle("Sample Patches — No Cancer (top) vs Cancer (bottom)", fontsize=13, fontweight="bold")

for col, img_id in enumerate(NEG_DISPLAY):
    img = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"{LABEL_NEG}\n{img_id[:8]}…", fontsize=7)
    axes[0, col].axis("off")

for col, img_id in enumerate(POS_DISPLAY):
    img = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")
    axes[1, col].imshow(img)
    axes[1, col].set_title(f"{LABEL_POS}\n{img_id[:8]}…", fontsize=7)
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  4. Class Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. CLASS DISTRIBUTION")
print("=" * 55)

counts = df["label"].value_counts().sort_index()  # sort_index → [0, 1] order
total  = len(df)
print(f"  {LABEL_NEG} : {counts[0]:,}  ({counts[0]/total*100:.1f}%)")
print(f"  {LABEL_POS}    : {counts[1]:,}  ({counts[1]/total*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart gives an absolute count view; pie chart shows proportions at a glance
axes[0].bar([LABEL_NEG, LABEL_POS], counts.values, color=["#4C9BE8", "#E8714C"], edgecolor="white")
axes[0].set_title("Class Distribution (counts)")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    # Annotate each bar with its exact count
    axes[0].text(i, v + 15, f"{v:,}", ha="center", fontsize=10)

axes[1].pie(
    counts.values,
    labels=[LABEL_NEG, LABEL_POS],
    colors=["#4C9BE8", "#E8714C"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white"),
)
axes[1].set_title("Class Distribution (%)")
plt.suptitle(f"Class Balance — {SAMPLES_PER_CLASS:,} per class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# Load all images into memory — needed for sections 7, 8, and the summary
# At 96×96×3 bytes each: 2 000 × 27 648 bytes ≈ 53 MB — well within RAM.
print("\n  Loading all 2 000 images …", end=" ", flush=True)
neg_arrays = load_images(neg_ids)
pos_arrays = load_images(pos_ids)
neg_flat   = np.stack(neg_arrays).reshape(-1, 3)   # (N*96*96, 3) uint8 — used in summary
pos_flat   = np.stack(pos_arrays).reshape(-1, 3)
channels   = ["R", "G", "B"]
print("done")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Average Image (mean pixel across all images in each class)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  7. AVERAGE IMAGE PER CLASS")
print("=" * 55)

# The mean image blurs out tissue variation and reveals the underlying colour
# bias of each class — useful for spotting systematic staining differences
neg_mean_img = np.stack(neg_arrays).mean(axis=0).astype(np.uint8)  # (96, 96, 3)
pos_mean_img = np.stack(pos_arrays).mean(axis=0).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(7, 4))
axes[0].imshow(neg_mean_img)
axes[0].set_title(f"Mean Image — {LABEL_NEG}\n(avg of {SAMPLES_PER_CLASS:,} patches)", fontsize=10)
axes[0].axis("off")

axes[1].imshow(pos_mean_img)
axes[1].set_title(f"Mean Image — {LABEL_POS}\n(avg of {SAMPLES_PER_CLASS:,} patches)", fontsize=10)
axes[1].axis("off")

plt.suptitle("Average Patch per Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_average_image.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  8. Pixel Brightness Distribution (grayscale proxy: mean of R, G, B)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  8. IMAGE BRIGHTNESS DISTRIBUTION")
print("=" * 55)

# Collapse H, W, and C into a single scalar per image.
# This is a fast proxy for perceived luminance without needing a proper
# luma conversion (e.g. 0.299R + 0.587G + 0.114B).
neg_brightness = np.stack(neg_arrays).mean(axis=(1, 2, 3))   # shape: (N,)
pos_brightness = np.stack(pos_arrays).mean(axis=(1, 2, 3))

print(f"  No-Cancer brightness — mean: {neg_brightness.mean():.2f}, std: {neg_brightness.std():.2f}")
print(f"  Cancer    brightness — mean: {pos_brightness.mean():.2f}, std: {pos_brightness.std():.2f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(neg_brightness, bins=40, alpha=0.65, density=True,
        color="#4C9BE8", label=LABEL_NEG, edgecolor="white")
ax.hist(pos_brightness, bins=40, alpha=0.65, density=True,
        color="#E8714C", label=LABEL_POS, edgecolor="white")

# Dashed vertical lines mark each class mean so the shift is immediately visible
ax.axvline(neg_brightness.mean(), color="#4C9BE8", linestyle="--", linewidth=1.5,
           label=f"{LABEL_NEG} mean={neg_brightness.mean():.1f}")
ax.axvline(pos_brightness.mean(), color="#E8714C", linestyle="--", linewidth=1.5,
           label=f"{LABEL_POS} mean={pos_brightness.mean():.1f}")
ax.set_xlabel("Mean Pixel Brightness (0–255)")
ax.set_ylabel("Density")
ax.set_title("Per-Image Brightness Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_brightness_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  9. Data Augmentation Preview
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  9. DATA AUGMENTATION PREVIEW")
print("=" * 55)

# Mirror the transforms used in training (data_loader.py) so this plot shows
# exactly what the model will see during training, not hypothetical transforms.
# p=1.0 forces both flips on so every augmented view looks clearly different.
augment = T.Compose([
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.RandomRotation(90),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
])

demo_ids = pos_ids[:3]   # 3 cancer patches — one per row
n_aug = 5                # column 0 = original, columns 1–4 = augmented views

fig, axes = plt.subplots(len(demo_ids), n_aug, figsize=(14, 7))
fig.suptitle("Augmentation Preview — Cancer Patches (each row = one original + 4 augmented views)",
             fontsize=11, fontweight="bold")

for row, img_id in enumerate(demo_ids):
    original = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")

    # Column 0: unmodified original for visual reference
    axes[row, 0].imshow(original)
    axes[row, 0].set_title("Original", fontsize=8)
    axes[row, 0].axis("off")

    # Columns 1–4: independently sampled augmented versions of the same patch
    for col in range(1, n_aug):
        aug_img = augment(original)
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f"Aug {col}", fontsize=8)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_augmentation_preview.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  10. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EXPLORATION SUMMARY")
print("=" * 55)
print(f"  Subset size        : {len(df):,} images  ({SAMPLES_PER_CLASS:,} per class)")
print(f"  Image dimensions   : 96 × 96 × 3 (RGB)")
print(f"  Label balance      : {counts[0]:,} no-cancer / {counts[1]:,} cancer  (50/50)")
print(f"\n  Dataset-wide stats (this subset):")

# Combine both classes to get the overall dataset statistics —
# these are the values you would use to normalise if recomputing MEAN/STD
all_flat = np.concatenate([neg_flat, pos_flat], axis=0)  # (2N*96*96, 3)
for i, ch in enumerate(channels):
    print(f"    {ch} channel — mean: {all_flat[:,i].mean()/255:.4f}  std: {all_flat[:,i].std()/255:.4f}")

print("=" * 55)
