# Histopathologic Cancer Detection — Group 5

## Group Members
Eric Emiowei: Data Engineer (Data Preparation Lead) 

---

## Dataset

- **Name:** Histopathologic Cancer Detection
- **Source:** https://www.kaggle.com/c/histopathologic-cancer-detection/data
- **Task:** Binary image classification — cancer (1) or no cancer (0)
- **Images:** 220,025 labelled tissue patches, 96×96 pixels, RGB, `.tif` format
- **Labels:** `train_labels.csv` — `id` (filename stem) | `label` (0 = No Cancer, 1 = Cancer)
- **Class balance:** 59.5% No Cancer (130,908) | 40.5% Cancer (89,117)

---

## How to Download the Dataset

**Step 1 — Get your Kaggle API key**

Go to [kaggle.com](https://kaggle.com) → profile icon → Settings → API → **Create New Token**

This downloads a `kaggle.json` file. Move it to the right location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Step 2 — Accept the competition rules**

Visit https://kaggle.com/c/histopathologic-cancer-detection, scroll down and click **"I Understand and Accept"**. This must be done manually before downloading.

**Step 3 — Download and unzip**

```bash
pip install kaggle
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip -d data/
```
## Project Overview
This project performs binary classification (tumor vs normal) on the Histopathologic Cancer Detection dataset (96×96 RGB patches).
I implemented and compared:

Unsupervised feature extraction with a CNN Autoencoder (latent vectors)
Transfer learning using ResNet50 as a frozen feature extractor (deep features) + optional PCA compression
Baseline CNN trained from scratch end-to-end on images + labels
Supervised custom CNN using the same PyTorch project structure as the unsupervised pipeline

## Metrics reported:

Accuracy
Precision / Recall / F1-score
ROC-AUC
---

## Project Folder Structure

```
project/
├── src/
│   └── data_loader.py 
        unsupervised/
      __main__.py
      runner.py
      datasets.py
      models.py
      train.py
      utils.py
         ← Data pipeline script
├── data/
│   ├── train/                  ← 220,025 .tif images go here
│   └── train_labels.csv        ← label file (id | label)
└── results/
    └── screenshots/            ← auto-created when script runs
        ├── class_distribution.png
        ├── split_distribution.png
        └── sample_images.png
``` └── pipeline/

---

## Dependencies

Install all required libraries:

```bash
pip install torch torchvision pillow pandas numpy matplotlib scikit-learn
```
Notes:

GPU is optional. For CUDA acceleration, install a CUDA-enabled PyTorch build.
On first use, pretrained ResNet50 weights may download into the local PyTorch cache
---

## How to Run

```bash
python src/data_loader.py
```
2) Unsupervised + transfer learning pipeline
Run as a module from inside src/:

cd Histopathologic-Cancer-Detection/src
python -m unsupervised

3) Supervised custom CNN
Run as a module from inside `src/`:

```bash
cd Histopathologic-Cancer-Detection/src
python3 -m supervised
```

Run the prepared supervised experiment set:

```bash
python3 src/supervised_experiments.py
```
---

## What the Script Does

1. Loads `train_labels.csv` and validates the dataset path
2. Splits data into **Train (75%) / Val (15%) / Test (10%)** — stratified to preserve class balance
3. Applies augmentation transforms for training (flips, rotation, colour jitter)
4. Applies normalisation-only transforms for val and test
5. Creates PyTorch `DataLoader` objects ready for model training
6. Verifies one batch shape and pixel range
7. Saves 3 screenshots to `results/screenshots/`

---
### What the Pipeline Does

1) **Create / load cached splits**  
- `results/pipeline/splits.npz`

2) **Autoencoder (unsupervised feature extraction)**  
- Trains a CNN autoencoder using images only (no labels)  
- Latent features:
  - `results/pipeline/autoencoder/features/ae_latents_train.npz`
  - `results/pipeline/autoencoder/features/ae_latents_val.npz`
  - `results/pipeline/autoencoder/features/ae_latents_test.npz`
- Best checkpoint:
  - `results/pipeline/autoencoder/checkpoints/autoencoder_best.pt`
- Training history:
  - `results/pipeline/logs/ae_history.csv`

3) **Transfer learning (ResNet50 feature extraction, frozen)**  
- PCA model:
  - `results/pipeline/transfer_learning/features/resnet50_pca256.joblib`
- PCA features:
  - `results/pipeline/transfer_learning/features/resnet50_pca256_train.npz`
  - `results/pipeline/transfer_learning/features/resnet50_pca256_val.npz`
  - `results/pipeline/transfer_learning/features/resnet50_pca256_test.npz`

4) **Baseline CNN (from scratch)**  
- Best checkpoint:
  - `results/pipeline/baseline_cnn/checkpoints/baseline_best.pt`
- Training history:
  - `results/pipeline/logs/cnn_history.csv`

5) **Supervised custom CNN**  
- Best checkpoint:
  - `results/supervised_cnn/checkpoints/<run_name>/custom_cnn_best.pt`
- Training history:
  - `results/supervised_cnn/logs/<run_name>_history.csv`
- Metrics:
  - `results/supervised_cnn/tables/metrics.csv`
- Experiment summary:
  - `report/supervised_experiments.csv`

6) **Metrics + plots**  
- Metrics table:
  - `results/pipeline/metrics.csv`
- Plots:
  - `results/pipeline/plots/comparison_test.png`
  - `results/pipeline/plots/roc_curves_test.png`

Notes on Re-running
The pipeline caches splits, features, and best checkpoints. If cached files exist, it skips expensive steps unless you enable the “force re-extract” settings inside src/unsupervised/runner.py.



## Output Screenshots

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart and pie chart showing cancer vs no-cancer image counts |
| `split_distribution.png` | Class balance preserved across train, val, and test splits |
| `sample_images.png` | 4 no-cancer (top) and 4 cancer (bottom) sample tissue patches |

---
### Results (Example Output)
After running the pipeline, the terminal prints metrics for each method and saves a summary table to:

- `results/pipeline/metrics.csv`

The project compares:
- **Baseline CNN** — final test metrics printed in terminal
- **Supervised custom CNN** — configurable custom architecture trained from scratch
- **Autoencoder latent features + classifier** — val/test metrics printed in terminal
- **ResNet50 (frozen) PCA features + classifier** — val/test metrics printed in terminal

The pipeline also saves comparison visuals to:
- `results/pipeline/plots/comparison_test.png` (bar chart comparison)
- `results/pipeline/plots/roc_curves_test.png` (ROC curves)

### Supervised CNN Experiment Controls

The custom CNN explicitly supports the required supervised-learning experiments:

- **Layers and filters:** change the `filters` tuple in `SupervisedConfig`
- **Dropout:** change `dropout` in `SupervisedConfig`
- **BatchNorm:** change `use_batchnorm` in `SupervisedConfig`
- **Learning rate:** change `learning_rate` in `SupervisedConfig`

Prepared experiment settings are included in `src/supervised_experiments.py`.


## How Other Members Use the Pipeline

Other team members can import the DataLoaders directly — no need to rewrite any data loading code:

```python
from data_loader import train_loader, val_loader, test_loader

# Each batch returns:
# images → torch.Tensor [32, 3, 96, 96]  (batch, channels, height, width)
# labels → torch.Tensor [32]              (0 = No Cancer, 1 = Cancer)

for images, labels in train_loader:
    # pass to your model here
    pass
```

**Note for Member 5 (EfficientNet):** EfficientNet expects 224×224 images. Override the transform in your script:

```python
from data_loader import train, val, test, MEAN, STD, CancerDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader

efficientnet_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

train_loader = DataLoader(
    CancerDataset(train["id"].tolist(), train["label"].tolist(), efficientnet_transforms),
    batch_size=32, shuffle=True, num_workers=4
)
```
clear
