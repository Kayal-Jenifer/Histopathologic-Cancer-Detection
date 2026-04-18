"""
Train SOTA — EfficientNetB0 Fine-Tuning for Histopathologic Cancer Detection
=============================================================================
Team Member : Arjun Mungath
Role        : State-of-the-Art (SOTA) Model

This is the main training entry point. It:
  1. Loads the same dataset and splits as the team's CNN pipeline
  2. Resizes images to 224x224 (EfficientNet requirement)
  3. Runs Phase 1: frozen base training (head only)
  4. Runs Phase 2: fine-tuning top EfficientNet layers
  5. Evaluates on the held-out test set
  6. Saves all models, logs, graphs, and metrics

Usage:
  python train_sota.py --csv ../../data/train_labels.csv --image_dir ../../data/train
  python train_sota.py --csv ../../data/train_labels.csv --image_dir ../../data/train --sample_size 20000
  python train_sota.py --csv ../../data/train_labels.csv --image_dir ../../data/train --sample_size 4000  # quick test on CPU

Outputs saved to results_sota/:
  models/       best_sota_phase1.keras, best_sota_phase2.keras, final_sota.keras
  logs/         sota_phase1_log.csv, sota_phase2_log.csv, history.csv
  graphs/       sota_accuracy.png, sota_loss.png
  tables/       sota_metrics.json, sota_test_results.csv
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from src.data_loader import load_labels, split_labels
from src.sota_model import SOTAConfig, build_efficientnet, compile_model, unfreeze_top_layers, get_callbacks


# -- Reproducibility -----------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -- Data pipeline for EfficientNet (224x224) ----------------------------------

AUTOTUNE = tf.data.AUTOTUNE

def _load_tif_image(filepath_bytes):
    """Load a .tif image using PIL (since TensorFlow can't read TIF natively)."""
    from PIL import Image
    import io
    filepath = filepath_bytes.numpy().decode("utf-8")
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def _parse_image_efficientnet(filename, label, image_dir, augment=False):
    """
    Load and preprocess one image for EfficientNet.

    Uses PIL to read .tif files (TensorFlow only supports JPEG/PNG/GIF/BMP/WebP).
    Resizes to 224x224 as required by EfficientNet.
    """
    image_path = tf.strings.join([image_dir, "/", filename])

    # Use tf.py_function to call PIL-based loader for TIF support
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
    """Create a tf.data pipeline resized to 224x224 for EfficientNet."""
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


# -- Plotting ------------------------------------------------------------------

def save_history_plots(phase1_df, phase2_df, output_dir):
    """Save accuracy and loss curves for both training phases."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Combine both phases for continuous plotting
    total_epochs_p1 = len(phase1_df)
    phase2_df_shifted = phase2_df.copy()
    phase2_df_shifted.index = phase2_df_shifted.index + total_epochs_p1

    # -- Accuracy plot --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase 1
    axes[0].plot(phase1_df.index + 1, phase1_df["accuracy"], "b-", label="Train Acc (Phase 1)")
    axes[0].plot(phase1_df.index + 1, phase1_df["val_accuracy"], "b--", label="Val Acc (Phase 1)")
    # Phase 2
    axes[0].plot(phase2_df_shifted.index + 1, phase2_df_shifted["accuracy"], "r-", label="Train Acc (Phase 2)")
    axes[0].plot(phase2_df_shifted.index + 1, phase2_df_shifted["val_accuracy"], "r--", label="Val Acc (Phase 2)")
    axes[0].axvline(x=total_epochs_p1 + 0.5, color="gray", linestyle=":", label="Unfreeze point")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("EfficientNet — Accuracy vs Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # -- Loss plot --
    axes[1].plot(phase1_df.index + 1, phase1_df["loss"], "b-", label="Train Loss (Phase 1)")
    axes[1].plot(phase1_df.index + 1, phase1_df["val_loss"], "b--", label="Val Loss (Phase 1)")
    axes[1].plot(phase2_df_shifted.index + 1, phase2_df_shifted["loss"], "r-", label="Train Loss (Phase 2)")
    axes[1].plot(phase2_df_shifted.index + 1, phase2_df_shifted["val_loss"], "r--", label="Val Loss (Phase 2)")
    axes[1].axvline(x=total_epochs_p1 + 0.5, color="gray", linestyle=":", label="Unfreeze point")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("EfficientNet — Loss vs Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("EfficientNetB0 SOTA — Two-Phase Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output / "sota_accuracy_loss.png", dpi=300)
    plt.close()
    print(f"  Saved training curves to {output / 'sota_accuracy_loss.png'}")


def save_roc_curve(y_true, y_prob, output_dir):
    """Save ROC curve plot for the SOTA model."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"EfficientNetB0 (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("EfficientNetB0 SOTA — ROC Curve (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "sota_roc_curve.png", dpi=300)
    plt.close()
    print(f"  Saved ROC curve to {output / 'sota_roc_curve.png'}")


def save_confusion_matrix_plot(y_true, y_pred, output_dir):
    """Save confusion matrix as a heatmap."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Cancer", "Cancer"])
    ax.set_yticklabels(["No Cancer", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("EfficientNetB0 SOTA — Confusion Matrix (Test Set)")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output / "sota_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"  Saved confusion matrix to {output / 'sota_confusion_matrix.png'}")


# -- Metrics -------------------------------------------------------------------

def save_metrics_report(y_true, y_prob, output_dir, threshold=0.5):
    """Save full metrics report (matches team's utils.save_metrics_report)."""
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


# -- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 SOTA for Histopathologic Cancer Detection"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to train_labels.csv")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train image folder")
    parser.add_argument("--output_dir", type=str, default="results_sota",
                        help="Folder to save models, logs, graphs, tables")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Total sample size (split equally per class). "
                             "Default: use full dataset. Use 4000 for quick CPU test.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-5)
    return parser.parse_args()


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()

    # Create output directories
    out = Path(args.output_dir)
    for sub in ["models", "logs", "graphs", "tables", "screenshots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Device info
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  GPU detected: {gpus[0].name}")
        # Prevent TF from grabbing all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  No GPU detected — running on CPU (will be slower)")

    # ── 1. Load and split data ───────────────────────────────────────────────
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

    # ── 2. Create datasets (224x224 for EfficientNet) ────────────────────────
    print("\n" + "=" * 60)
    print("  2. CREATING DATASETS (224x224)")
    print("=" * 60)

    train_ds = make_efficientnet_dataset(
        train_df, args.image_dir, batch_size=args.batch_size, shuffle=True, augment=True
    )
    val_ds = make_efficientnet_dataset(
        val_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False
    )
    test_ds = make_efficientnet_dataset(
        test_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False
    )

    # Verify one batch
    sample_imgs, sample_lbls = next(iter(train_ds))
    print(f"  Batch shape : {tuple(sample_imgs.shape)}")
    print(f"  Label shape : {tuple(sample_lbls.shape)}")
    print(f"  Pixel range : [{sample_imgs.numpy().min():.2f}, {sample_imgs.numpy().max():.2f}]")

    # ── 3. Build model ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  3. BUILDING EFFICIENTNETB0")
    print("=" * 60)

    config = SOTAConfig(
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_learning_rate=args.phase1_lr,
        phase2_learning_rate=args.phase2_lr,
    )
    model, base_model = build_efficientnet(config)
    model = compile_model(model, learning_rate=config.phase1_learning_rate)

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"  Trainable params (Phase 1) : {trainable:,}")
    print(f"  Frozen params              : {frozen:,}")
    print(f"  Base model layers          : {len(base_model.layers)}")

    # ── 4. Phase 1: Train classification head (base frozen) ──────────────────
    print("\n" + "=" * 60)
    print("  4. PHASE 1 — TRAINING HEAD (BASE FROZEN)")
    print("=" * 60)

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase1_epochs,
        callbacks=get_callbacks(args.output_dir, phase="phase1"),
        verbose=1,
    )
    phase1_df = pd.DataFrame(history_p1.history)
    phase1_df.index.name = "epoch"
    phase1_df.to_csv(out / "logs" / "sota_phase1_history.csv")

    p1_val_acc = max(history_p1.history["val_accuracy"])
    p1_val_auc = max(history_p1.history["val_auc"])
    print(f"\n  Phase 1 best — Val Acc: {p1_val_acc:.4f} | Val AUC: {p1_val_auc:.4f}")

    # ── 5. Phase 2: Fine-tune top EfficientNet layers ────────────────────────
    print("\n" + "=" * 60)
    print("  5. PHASE 2 — FINE-TUNING TOP LAYERS")
    print("=" * 60)

    model = unfreeze_top_layers(model, base_model, config)

    trainable_p2 = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable params (Phase 2) : {trainable_p2:,}")
    print(f"  Unfrozen from layer        : {config.unfreeze_from} / {len(base_model.layers)}")

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase2_epochs,
        callbacks=get_callbacks(args.output_dir, phase="phase2"),
        verbose=1,
    )
    phase2_df = pd.DataFrame(history_p2.history)
    phase2_df.index.name = "epoch"
    phase2_df.to_csv(out / "logs" / "sota_phase2_history.csv")

    p2_val_acc = max(history_p2.history["val_accuracy"])
    p2_val_auc = max(history_p2.history["val_auc"])
    print(f"\n  Phase 2 best — Val Acc: {p2_val_acc:.4f} | Val AUC: {p2_val_auc:.4f}")

    # ── 6. Save combined history ─────────────────────────────────────────────
    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    combined_df.index.name = "epoch"
    combined_df.to_csv(out / "logs" / "sota_full_history.csv")

    # ── 7. Evaluate on test set ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  6. EVALUATING ON TEST SET")
    print("=" * 60)

    y_prob = model.predict(test_ds).ravel()
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = (y_prob >= 0.5).astype(int)

    # Keras built-in metrics
    eval_results = model.evaluate(test_ds, verbose=0)
    eval_dict = dict(zip(model.metrics_names, eval_results))

    # Sklearn metrics
    metrics = save_metrics_report(y_true, y_prob, str(out / "tables"))

    # Save test results CSV (same format as team's CNN)
    pd.DataFrame([eval_dict]).to_csv(out / "tables" / "sota_test_results.csv", index=False)

    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Cancer", "Cancer"]))

    print("  === Final Test Metrics ===")
    for key, value in eval_dict.items():
        print(f"  {key}: {value:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # ── 8. Save plots ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  7. SAVING PLOTS")
    print("=" * 60)

    save_history_plots(phase1_df, phase2_df, str(out / "graphs"))
    save_roc_curve(y_true, y_prob, str(out / "graphs"))
    save_confusion_matrix_plot(y_true, y_pred, str(out / "graphs"))

    # ── 9. Save final model ──────────────────────────────────────────────────
    model.save(out / "models" / "final_sota.keras")
    print(f"\n  Final model saved to {out / 'models' / 'final_sota.keras'}")

    # ── 10. Print summary for team comparison table ──────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY — Copy these into model_comparison_template.csv")
    print("=" * 60)
    cr = metrics["classification_report"]
    # Get accuracy from eval_dict (key name varies by TF version)
    test_acc = eval_dict.get("accuracy", eval_dict.get("compile_metrics", 0.0))
    # Get cancer class metrics (key could be '1.0' or '1')
    cancer_key = '1.0' if '1.0' in cr else '1'
    print(f"  model_name    : EfficientNetB0_SOTA")
    print(f"  train_samples : {len(train_df)}")
    print(f"  val_samples   : {len(val_df)}")
    print(f"  test_samples  : {len(test_df)}")
    print(f"  accuracy      : {test_acc:.4f}")
    print(f"  precision     : {cr[cancer_key]['precision']:.4f}")
    print(f"  recall        : {cr[cancer_key]['recall']:.4f}")
    print(f"  f1_score      : {cr[cancer_key]['f1-score']:.4f}")
    print(f"  roc_auc       : {metrics['roc_auc']:.4f}")

    print("\n  Done! All outputs saved to:", out.resolve())


if __name__ == "__main__":
    main()
