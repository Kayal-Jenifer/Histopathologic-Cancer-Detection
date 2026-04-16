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
## Unsupervised Learning (Feature Extraction)

Our unsupervised component is implemented in `src/unsupervised.py` and focuses on **feature extraction**:

- Train an unsupervised **denoising convolutional autoencoder (CAE)** on unlabeled images.
- Use the encoder bottleneck as a learned feature vector per patch.
- Run a **transfer learning** experiment to analyze how **pre-training** affects supervised performance on the same cancer vs no-cancer labels.

---



## Metrics reported:

`src/unsupervised.py` reports:

- **Autoencoder reconstruction MSE** per epoch (and the **best epoch** by lowest MSE).
- **Transfer learning supervised metrics** using Logistic Regression on frozen ResNet18 features:
  - **Accuracy**
  - **ROC-AUC**

---

## Project Folder Structure

```
Histopathologic-Cancer-Detection/
├── src/
│   ├── data_exploration.py
│   ├── supervised_learning.py
│   ├── unsupervised.py
│   └── newreadme.md
├── data/
│   ├── train/               ← .tif images
│   └── train_labels.csv     ← labels (id, label)
└── results/
    └── screenshots/         ← plots saved by scripts
```

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
## How to Run (Unsupervised)
From `Histopathologic-Cancer-Detection/`:

```bash
python src/unsupervised.py
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
## What `src/unsupervised.py` Does

1. Loads a balanced labeled subset (1,000 cancer + 1,000 no-cancer) from `data/train_labels.csv`.
2. Trains a denoising CAE and prints epoch-wise reconstruction MSE + best epoch.
3. Extracts and prints the encoder feature shape (sanity check for feature extraction).
4. Runs transfer-learning evaluation on the supervised labels:
   - ResNet18 features without pre-training vs ImageNet pre-trained ResNet18
   - Reports Accuracy and ROC-AUC.
5. Saves performance visuals to `results/screenshots/`.

---

## Output Screenshots

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart and pie chart showing cancer vs no-cancer image counts |
| `split_distribution.png` | Class balance preserved across train, val, and test splits |
| `sample_images.png` | 4 no-cancer (top) and 4 cancer (bottom) sample tissue patches |

---
## Unsupervised Output Screenshots

| File | Description |
|------|-------------|
| `results/screenshots/unsupervised_ae_mse.png` | Autoencoder training curve (reconstruction MSE vs epoch) |
| `results/screenshots/unsupervised_transfer_learning.png` | Bar chart comparing no pre-training vs ImageNet pre-training (Accuracy, ROC-AUC) |
### Results (Example Output)


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
