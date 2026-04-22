"""
Train SOTA — EfficientNetB0 for Histopathologic Cancer Detection
Runs both pretrained (ImageNet) and from-scratch versions, then compares.

Usage:
  python train_sota.py --csv data/train_labels.csv --image_dir data/train
  python train_sota.py --csv data/train_labels.csv --image_dir data/train --sample_size 20000
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from src.sota_model import SOTAConfig, build_efficientnet, compile_model, unfreeze_top_layers, get_callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df["filename"] = df["id"].astype(str) + ".tif"
    return df


def split_labels(df, val_size=0.15, test_size=0.10, random_state=42):
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, stratify=df["label"], random_state=random_state)
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_tif_image(filepath_bytes):
    from PIL import Image
    filepath = filepath_bytes.numpy().decode("utf-8")
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    return np.array(img, dtype=np.float32) / 255.0


def _parse_image_efficientnet(filename, label, image_dir, augment=False):
    image_path = tf.strings.join([image_dir, "/", filename])
    image = tf.py_function(_load_tif_image, [image_path], tf.float32)
    image.set_shape([224, 224, 3])

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.10)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    label = tf.cast(label, tf.float32)
    return image, label


def make_efficientnet_dataset(df, image_dir, batch_size=32, shuffle=False, augment=False):
    image_dir = str(Path(image_dir))
    ds = tf.data.Dataset.from_tensor_slices((df["filename"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: _parse_image_efficientnet(x, y, image_dir=image_dir, augment=augment),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def save_metrics_report(y_true, y_prob, output_dir, threshold=0.5):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    with open(output / "sota_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def save_plots(phase1_df, phase2_df, y_true, y_prob, y_pred, output_dir, label="pretrained"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Accuracy and loss curves
    total_epochs_p1 = len(phase1_df)
    phase2_df_shifted = phase2_df.copy()
    phase2_df_shifted.index = phase2_df_shifted.index + total_epochs_p1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(phase1_df.index + 1, phase1_df["accuracy"], "b-", label="Train Acc (Phase 1)")
    axes[0].plot(phase1_df.index + 1, phase1_df["val_accuracy"], "b--", label="Val Acc (Phase 1)")
    axes[0].plot(phase2_df_shifted.index + 1, phase2_df_shifted["accuracy"], "r-", label="Train Acc (Phase 2)")
    axes[0].plot(phase2_df_shifted.index + 1, phase2_df_shifted["val_accuracy"], "r--", label="Val Acc (Phase 2)")
    axes[0].axvline(x=total_epochs_p1 + 0.5, color="gray", linestyle=":", label="Unfreeze point")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"EfficientNet ({label}) — Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(phase1_df.index + 1, phase1_df["loss"], "b-", label="Train Loss (Phase 1)")
    axes[1].plot(phase1_df.index + 1, phase1_df["val_loss"], "b--", label="Val Loss (Phase 1)")
    axes[1].plot(phase2_df_shifted.index + 1, phase2_df_shifted["loss"], "r-", label="Train Loss (Phase 2)")
    axes[1].plot(phase2_df_shifted.index + 1, phase2_df_shifted["val_loss"], "r--", label="Val Loss (Phase 2)")
    axes[1].axvline(x=total_epochs_p1 + 0.5, color="gray", linestyle=":", label="Unfreeze point")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"EfficientNet ({label}) — Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"EfficientNetB0 — {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output / f"sota_{label}_accuracy_loss.png", dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"EfficientNetB0 {label} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"EfficientNetB0 ({label}) — ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / f"sota_{label}_roc_curve.png", dpi=300)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Cancer", "Cancer"])
    ax.set_yticklabels(["No Cancer", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"EfficientNetB0 ({label}) — Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output / f"sota_{label}_confusion_matrix.png", dpi=300)
    plt.close()


def run_experiment(config, train_ds, val_ds, test_ds, output_dir, pretrained=True):
    label = "pretrained" if pretrained else "from_scratch"
    out = Path(output_dir)

    print("\n" + "=" * 60)
    print(f"  BUILDING EFFICIENTNETB0 ({label.upper()})")
    print("=" * 60)

    model, base_model = build_efficientnet(config, pretrained=pretrained)
    model = compile_model(model, learning_rate=config.phase1_learning_rate)

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"  Pretrained       : {pretrained}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}")

    # Phase 1
    print(f"\n  PHASE 1 — {'TRAINING HEAD (BASE FROZEN)' if pretrained else 'TRAINING ALL LAYERS'}")

    history_p1 = model.fit(
        train_ds, validation_data=val_ds, epochs=config.phase1_epochs,
        callbacks=get_callbacks(output_dir, phase=f"{label}_phase1"), verbose=1,
    )
    phase1_df = pd.DataFrame(history_p1.history)
    phase1_df.to_csv(out / "logs" / f"sota_{label}_phase1_history.csv")

    print(f"\n  Phase 1 best — Val Acc: {max(history_p1.history['val_accuracy']):.4f} | Val AUC: {max(history_p1.history['val_auc']):.4f}")

    # Phase 2
    if pretrained:
        print(f"\n  PHASE 2 — FINE-TUNING TOP LAYERS")
        model = unfreeze_top_layers(model, base_model, config)
    else:
        print(f"\n  PHASE 2 — CONTINUED TRAINING (ALL LAYERS)")
        compile_model(model, learning_rate=config.phase2_learning_rate)

    history_p2 = model.fit(
        train_ds, validation_data=val_ds, epochs=config.phase2_epochs,
        callbacks=get_callbacks(output_dir, phase=f"{label}_phase2"), verbose=1,
    )
    phase2_df = pd.DataFrame(history_p2.history)
    phase2_df.to_csv(out / "logs" / f"sota_{label}_phase2_history.csv")

    print(f"\n  Phase 2 best — Val Acc: {max(history_p2.history['val_accuracy']):.4f} | Val AUC: {max(history_p2.history['val_auc']):.4f}")

    # Evaluate on test set
    print(f"\n  EVALUATING ({label.upper()}) ON TEST SET")

    y_prob = model.predict(test_ds).ravel()
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = (y_prob >= 0.5).astype(int)

    eval_results = model.evaluate(test_ds, verbose=0)
    eval_dict = dict(zip(model.metrics_names, eval_results))
    metrics = save_metrics_report(y_true, y_prob, str(out / "tables"))

    print(f"\n  Classification Report ({label}):")
    print(classification_report(y_true, y_pred, target_names=["No Cancer", "Cancer"]))

    # Save plots
    save_plots(phase1_df, phase2_df, y_true, y_prob, y_pred, str(out / "graphs"), label=label)

    # Save model
    model.save(out / "models" / f"final_sota_{label}.keras")

    # Get summary metrics
    test_acc = eval_dict.get("accuracy", eval_dict.get("compile_metrics", 0.0))
    cr = metrics["classification_report"]
    cancer_key = '1.0' if '1.0' in cr else '1'

    return {
        "model": label,
        "accuracy": test_acc,
        "precision": cr[cancer_key]["precision"],
        "recall": cr[cancer_key]["recall"],
        "f1_score": cr[cancer_key]["f1-score"],
        "roc_auc": metrics["roc_auc"],
    }


def save_comparison(results, output_dir):
    out = Path(output_dir)

    # Comparison table
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(out / "tables" / "sota_comparison.csv", index=False)

    # Comparison bar chart
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, [results[0][m] for m in metrics_to_plot], width, label="Pretrained (ImageNet)", color="#4C9BE8")
    bars2 = ax.bar(x + width / 2, [results[1][m] for m in metrics_to_plot], width, label="From Scratch (Random)", color="#E8714C")

    ax.set_ylabel("Score")
    ax.set_title("EfficientNetB0 — Pretrained vs From Scratch")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out / "graphs" / "sota_pretrained_vs_scratch.png", dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 SOTA")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_sota")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    out = Path(args.output_dir)
    for sub in ["models", "logs", "graphs", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  No GPU detected — running on CPU")

    # 1. Load and split data
    print("\n" + "=" * 60)
    print("  1. LOADING DATA")
    print("=" * 60)

    df = load_labels(args.csv)
    if args.sample_size is not None:
        neg = df[df["label"] == 0].sample(min(len(df[df["label"] == 0]), args.sample_size // 2), random_state=SEED)
        pos = df[df["label"] == 1].sample(min(len(df[df["label"] == 1]), args.sample_size // 2), random_state=SEED)
        df = pd.concat([neg, pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} images ({args.sample_size // 2:,} per class)")

    train_df, val_df, test_df = split_labels(df)
    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}")

    # 2. Create datasets
    print("\n" + "=" * 60)
    print("  2. CREATING DATASETS (224x224)")
    print("=" * 60)

    train_ds = make_efficientnet_dataset(train_df, args.image_dir, batch_size=args.batch_size, shuffle=True, augment=True)
    val_ds = make_efficientnet_dataset(val_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False)
    test_ds = make_efficientnet_dataset(test_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False)

    sample_imgs, sample_lbls = next(iter(train_ds))
    print(f"  Batch shape : {tuple(sample_imgs.shape)}")
    print(f"  Label shape : {tuple(sample_lbls.shape)}")

    config = SOTAConfig(
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_learning_rate=args.phase1_lr,
        phase2_learning_rate=args.phase2_lr,
    )

    results = []

    # 3. Run pretrained version
    print("\n" + "#" * 60)
    print("  EXPERIMENT 1: PRETRAINED (ImageNet weights)")
    print("#" * 60)
    r1 = run_experiment(config, train_ds, val_ds, test_ds, args.output_dir, pretrained=True)
    results.append(r1)

    # 4. Run from-scratch version
    print("\n" + "#" * 60)
    print("  EXPERIMENT 2: FROM SCRATCH (random weights)")
    print("#" * 60)
    r2 = run_experiment(config, train_ds, val_ds, test_ds, args.output_dir, pretrained=False)
    results.append(r2)

    # 5. Compare
    print("\n" + "=" * 60)
    print("  COMPARISON — PRETRAINED vs FROM SCRATCH")
    print("=" * 60)

    save_comparison(results, args.output_dir)

    print(f"\n  {'Metric':<15} {'Pretrained':>12} {'From Scratch':>14} {'Difference':>12}")
    print("  " + "-" * 55)
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        diff = r1[m] - r2[m]
        sign = "+" if diff > 0 else ""
        print(f"  {m:<15} {r1[m]:>12.4f} {r2[m]:>14.4f} {sign}{diff:>11.4f}")

    print(f"\n  Pre-training {'improved' if r1['roc_auc'] > r2['roc_auc'] else 'did not improve'} ROC-AUC by {abs(r1['roc_auc'] - r2['roc_auc']):.4f}")
    print(f"\n  Done! All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
