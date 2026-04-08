from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T

from supervised.datasets import LabeledImageDataset
from supervised.models import CustomCNN
from supervised.train import train_custom_cnn
from supervised.utils import append_metrics_row, compute_metrics, get_or_create_splits, make_loader, seed_everything


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT_DIR / "data" / "train"
LABELS_CSV = ROOT_DIR / "data" / "train_labels.csv"

RESULTS_DIR = ROOT_DIR / "results" / "supervised_cnn"
CKPT_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
SPLITS_PATH = RESULTS_DIR / "splits.npz"
METRICS_PATH = TABLES_DIR / "metrics.csv"
EXPERIMENTS_PATH = ROOT_DIR / "report" / "supervised_experiments.csv"

MEAN = (0.7008, 0.5384, 0.6916)
STD = (0.2350, 0.2774, 0.2128)


@dataclass(frozen=True)
class SupervisedConfig:
    seed: int = 42
    reuse_splits: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    filters: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.4
    use_batchnorm: bool = True
    epochs: int = 15
    learning_rate: float = 1e-3
    batch_size: int = 64


def main() -> None:
    cfg = SupervisedConfig()
    run_supervised_pipeline(cfg, run_name="baseline")


def run_supervised_pipeline(cfg: SupervisedConfig, run_name: str) -> dict:
    seed_everything(cfg.seed)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {TRAIN_DIR.resolve()}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV.resolve()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    print("Split sizes:", {key: len(value[0]) for key, value in splits.items()})

    train_transform = T.Compose(
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
    eval_transform = T.Compose([T.Resize((96, 96)), T.ToTensor(), T.Normalize(MEAN, STD)])

    train_dataset = LabeledImageDataset(TRAIN_DIR, splits["train"][0], splits["train"][1], train_transform)
    val_dataset = LabeledImageDataset(TRAIN_DIR, splits["val"][0], splits["val"][1], eval_transform)
    test_dataset = LabeledImageDataset(TRAIN_DIR, splits["test"][0], splits["test"][1], eval_transform)

    num_workers = 0
    train_loader = make_loader(train_dataset, batch_size=cfg.batch_size, num_workers=num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=cfg.batch_size, num_workers=num_workers, shuffle=False)
    test_loader = make_loader(test_dataset, batch_size=cfg.batch_size, num_workers=num_workers, shuffle=False)

    model = CustomCNN(filters=cfg.filters, dropout=cfg.dropout, use_batchnorm=cfg.use_batchnorm).to(device)
    best_path = train_custom_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        ckpt_dir=CKPT_DIR / run_name,
        log_csv=LOGS_DIR / f"{run_name}_history.csv",
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_prob = _evaluate_classifier(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_prob)
    result = {
        "run_name": run_name,
        "filters": ",".join(str(value) for value in cfg.filters),
        "dropout": cfg.dropout,
        "use_batchnorm": cfg.use_batchnorm,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        **test_metrics,
    }
    append_metrics_row(METRICS_PATH, result)
    append_metrics_row(EXPERIMENTS_PATH, result)
    _save_probability_plot(y_true, y_prob, PLOTS_DIR / f"{run_name}_probabilities.png")
    print("Test metrics:", test_metrics)
    return result


@torch.inference_mode()
def _evaluate_classifier(model, loader, device):
    model.eval()
    labels = []
    probs = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype("float32")
        probs.extend(batch_probs.tolist())
        labels.extend(int(target) for target in targets)
    return pd.Series(labels, dtype="int64").to_numpy(), pd.Series(probs, dtype="float32").to_numpy()


def _save_probability_plot(y_true, y_prob, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="No Cancer")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Cancer")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted cancer probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
