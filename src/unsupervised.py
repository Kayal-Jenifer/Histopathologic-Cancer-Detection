"""
Unsupervised Learning — Histopathologic Cancer Detection

"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"
SCREENSHOTS_DIR = ROOT_DIR / "results" / "screenshots"

SEED = 42
SAMPLES_PER_CLASS = 1_000
IMAGE_SIZE = 96


# unsupervised representation learner (AE)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NOISE_STD = 0.05
LATENT_CHANNELS = 256

#standardize image
TL_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUP_TEST_SIZE = 0.20


def seed_everything(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_existing_df() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    existing = {p.stem for p in TRAIN_DIR.glob("*.tif")}
    return df[df["id"].isin(existing)].reset_index(drop=True)


def sample_balanced_ids() -> list[str]:
    # Use an equal number of cancer / no-cancer patches so the learned features
    # aren't dominated by the majority class distribution.
    df = _load_existing_df()
    neg = df[df["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=SEED)
    pos = df[df["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=SEED)
    subset = pd.concat([neg, pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return subset["id"].astype(str).tolist()


def sample_balanced_df() -> pd.DataFrame:
    # Same balanced subset as IDs, but keep labels for supervised evaluation.
    df = _load_existing_df()
    neg = df[df["label"] == 0].sample(SAMPLES_PER_CLASS, random_state=SEED)
    pos = df[df["label"] == 1].sample(SAMPLES_PER_CLASS, random_state=SEED)
    return pd.concat([neg, pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def make_transfer_transform() -> T.Compose:
    # ResNet18 pre-training is on ImageNet, so we use ImageNet normalization.
    return T.Compose([T.Resize((TL_IMAGE_SIZE, TL_IMAGE_SIZE)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LabeledDataset(Dataset):
    # Minimal dataset for supervised evaluation: returns (image, label).
    def __init__(self, df: pd.DataFrame, transform: T.Compose) -> None:
        self.ids = df["id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image = Image.open(TRAIN_DIR / f"{self.ids[idx]}.tif").convert("RGB")
        x = self.transform(image)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def make_transforms() -> tuple[T.Compose, T.Compose]:
    # We train a denoising AE: the model sees a corrupted view (noisy/augmented)
    # but must reconstruct the clean image, encouraging robust features.
    clean = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    noisy = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
        ]
    )
    return clean, noisy


class DenoisingDataset(Dataset):
    # Dataset returns (noisy_input, clean_target) for an unsupervised objective.
    def __init__(self, ids: list[str], clean_tfm: T.Compose, noisy_tfm: T.Compose, noise_std: float) -> None:
        self.ids = ids
        self.clean_tfm = clean_tfm
        self.noisy_tfm = noisy_tfm
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image = Image.open(TRAIN_DIR / f"{image_id}.tif").convert("RGB")

        clean = self.clean_tfm(image)
        noisy = self.noisy_tfm(image)
        if self.noise_std > 0:
            # Add Gaussian noise so the AE must learn structure, not pixel-copying.
            noisy = (noisy + torch.randn_like(noisy) * self.noise_std).clamp(0.0, 1.0)
        return noisy, clean


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        # Encoder compresses the 96x96 patch into a small spatial bottleneck.
        # That bottleneck is what we later treat as the "features".
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 96 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(inplace=True),
        )
        # Decoder mirrors the encoder to reconstruct the original image.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1),  # 6 -> 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 24 -> 48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 48 -> 96
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.inference_mode()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)


def train_autoencoder(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[int, float, list[float]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Training autoencoder (epoch-wise reconstruction MSE):")

    best_epoch = 0
    best_mse = float("inf")
    mse_history = []

    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        n = 0
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            recon = model(noisy)
            # MSE is a standard reconstruction loss for denoising AEs.
            loss = F.mse_loss(recon, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(noisy.size(0))
            total += float(loss.detach().cpu()) * bs
            n += bs

        epoch_mse = float(total / max(n, 1))
        mse_history.append(epoch_mse)
        print(f"Epoch {epoch:02d}/{EPOCHS}  mse={epoch_mse:.6f}")
        if epoch_mse < best_mse:
            # Track the best training epoch
            best_mse = epoch_mse
            best_epoch = epoch

    return best_epoch, float(best_mse), mse_history


def save_ae_mse_plot(mse_history: list[float]) -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(mse_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, mse_history, marker="o")
    plt.title("Autoencoder Training — Reconstruction MSE", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "unsupervised_ae_mse.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


@torch.inference_mode()
def extract_features(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    # Run a frozen feature extractor over a labeled dataset.
    model.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        feats.append(z.detach().cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def eval_logreg(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    # Simple supervised head to quantify how useful the extracted features are.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return float(accuracy_score(y_test, pred)), float(roc_auc_score(y_test, prob))


def run_transfer_learning(df: pd.DataFrame, device: torch.device) -> tuple[tuple[float, float], tuple[float, float]]:
    # Compare supervised performance: ResNet18 with vs without ImageNet pre-training.
    train_df, test_df = train_test_split(df, test_size=SUP_TEST_SIZE, stratify=df["label"], random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nTransfer learning (ResNet18 frozen features + LogisticRegression):")

    transform = make_transfer_transform()
    train_loader = DataLoader(
        LabeledDataset(train_df, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        LabeledDataset(test_df, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    metrics_no_pretrain = None
    metrics_pretrained = None

    for pretrained in (False, True):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone = backbone.to(device)

        x_train, y_train = extract_features(backbone, train_loader, device)
        x_test, y_test = extract_features(backbone, test_loader, device)
        acc, auc = eval_logreg(x_train, y_train, x_test, y_test)
        if pretrained:
            print(f"ImageNet pre-trained ResNet18: accuracy={acc:.4f}  roc_auc={auc:.4f}")
            metrics_pretrained = (acc, auc)
        else:
            print(f"ResNet18 features without pre-training: accuracy={acc:.4f}  roc_auc={auc:.4f}")
            metrics_no_pretrain = (acc, auc)

    return metrics_no_pretrain, metrics_pretrained


def save_transfer_learning_plot(metrics_no_pretrain: tuple[float, float], metrics_pretrained: tuple[float, float]) -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Accuracy", "ROC-AUC"]
    no_pre = [metrics_no_pretrain[0], metrics_no_pretrain[1]]
    pre = [metrics_pretrained[0], metrics_pretrained[1]]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, no_pre, width, label="No pre-training")
    plt.bar(x + width / 2, pre, width, label="ImageNet pre-trained")
    plt.title("Transfer Learning — ResNet18 Feature Quality", fontweight="bold")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "unsupervised_transfer_learning.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main() -> None:
    # Train the unsupervised AE, then run the transfer-learning comparison.
    seed_everything(SEED)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {TRAIN_DIR.resolve()}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = sample_balanced_ids()
    clean_tfm, noisy_tfm = make_transforms()
    dataset = DenoisingDataset(ids, clean_tfm, noisy_tfm, noise_std=NOISE_STD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = ConvAutoencoder().to(device)
    best_epoch, best_mse, mse_history = train_autoencoder(model, loader, device)
    print(f"Best epoch: {best_epoch}  best_mse={best_mse:.6f}")
    save_ae_mse_plot(mse_history)

    # Feature extraction (encoder bottleneck)
    model.eval()
    noisy, clean = next(iter(loader))
    # Demonstrate feature extraction on one batch; each image becomes a vector.
    features = model.extract_features(clean.to(device)).cpu()
    print(f"Extracted features: {tuple(features.shape)}")

    df = sample_balanced_df()
    metrics_no_pretrain, metrics_pretrained = run_transfer_learning(df, device)
    if metrics_no_pretrain is not None and metrics_pretrained is not None:
        save_transfer_learning_plot(metrics_no_pretrain, metrics_pretrained)


if __name__ == "__main__":
    main()
