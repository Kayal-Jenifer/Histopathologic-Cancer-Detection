from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ["--filters", "32,64,128", "--dropout", "0.3", "--epochs", "10", "--run_name", "cnn_baseline"],
    ["--filters", "32,64,128,256", "--dropout", "0.4", "--epochs", "12", "--run_name", "cnn_deeper"],
    ["--filters", "32,64,128,256", "--dropout", "0.5", "--epochs", "12", "--run_name", "cnn_regularized"],
]

for args in EXPERIMENTS:
    cmd = [sys.executable, str(ROOT / "train_cnn.py")] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)
