import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "hyperparameter_runs.csv"

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "filters", "dropout", "epochs", "learning_rate", "val_accuracy", "val_auc", "notes"])

print(f"Use this file to record and compare CNN experiments: {CSV_PATH}")
