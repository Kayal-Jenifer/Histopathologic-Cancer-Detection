import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from src.cnn_model import CNNConfig, build_custom_cnn, compile_model
from src.data_loader import load_labels, split_labels, make_dataset
from src.utils import ensure_dirs, HistoryLogger, save_history_plots, save_metrics_report


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom CNN for Histopathologic Cancer Detection")
    parser.add_argument("--csv", type=str, required=True, help="Path to train_labels.csv")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train image folder")
    parser.add_argument("--output_dir", type=str, default="results", help="Folder to save models, logs, graphs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--dense_units", type=int, default=256)
    parser.add_argument("--filters", type=str, default="32,64,128,256", help="Comma-separated filter sizes")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional sample size for faster experiments")
    return parser.parse_args()



def main():
    args = parse_args()
    ensure_dirs(args.output_dir)

    df = load_labels(args.csv)
    if args.sample_size is not None:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.sample_size // 2), random_state=SEED)
        ).reset_index(drop=True)

    train_df, val_df, test_df = split_labels(df)

    train_ds = make_dataset(train_df, args.image_dir, batch_size=args.batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(val_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False)
    test_ds = make_dataset(test_df, args.image_dir, batch_size=args.batch_size, shuffle=False, augment=False)

    filters = tuple(int(x.strip()) for x in args.filters.split(","))
    config = CNNConfig(conv_blocks=filters, dropout_rate=args.dropout, dense_units=args.dense_units)
    model = build_custom_cnn(config)
    model = compile_model(model, learning_rate=args.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(args.output_dir) / "models" / "best_cnn.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(Path(args.output_dir) / "logs" / "training_log.csv"), append=False),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = HistoryLogger(str(Path(args.output_dir) / "logs" / "history.csv")).save(history)
    save_history_plots(history_df, str(Path(args.output_dir) / "graphs"))

    y_prob = model.predict(test_ds).ravel()
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    metrics = save_metrics_report(y_true, y_prob, str(Path(args.output_dir) / "tables"))

    eval_results = model.evaluate(test_ds, verbose=0)
    eval_dict = dict(zip(model.metrics_names, eval_results))
    pd.DataFrame([eval_dict]).to_csv(Path(args.output_dir) / "tables" / "test_results.csv", index=False)

    model.save(Path(args.output_dir) / "models" / "final_cnn.keras")

    print("\n=== Final Test Metrics ===")
    for key, value in eval_dict.items():
        print(f"{key}: {value:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
