from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def ensure_dirs(base_dir: str):
    base = Path(base_dir)
    for sub in [
        "graphs",
        "logs",
        "models",
        "tables",
        "screenshots",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)


class HistoryLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def save(self, history):
        history_df = pd.DataFrame(history.history)
        history_df.index.name = "epoch"
        history_df.to_csv(self.csv_path, index=True)
        return history_df


def save_history_plots(history_df: pd.DataFrame, output_dir: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df.index + 1, history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df.index + 1, history_df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "cnn_accuracy.png", dpi=300)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df.index + 1, history_df["loss"], label="Train Loss")
    plt.plot(history_df.index + 1, history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "cnn_loss.png", dpi=300)
    plt.close()



def save_metrics_report(y_true, y_prob, output_dir: str, threshold: float = 0.5):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    with open(output / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
