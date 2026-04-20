"""SOTA Model — EfficientNetB0 for Histopathologic Cancer Detection."""

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class SOTAConfig:
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 1
    dropout_rate: float = 0.4
    dense_units: int = 256
    phase1_learning_rate: float = 1e-3
    phase1_epochs: int = 5
    phase2_learning_rate: float = 1e-5
    phase2_epochs: int = 10
    unfreeze_from: int = 200
    batch_size: int = 32
    l2_weight: float = 1e-4


def build_efficientnet(config: SOTAConfig):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=config.input_shape,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=config.input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_weight),
        name="head_dense",
    )(x)
    x = layers.Dropout(config.dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_SOTA")
    return model, base_model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(model, base_model, config):
    base_model.trainable = True
    for layer in base_model.layers[:config.unfreeze_from]:
        layer.trainable = False
    compile_model(model, learning_rate=config.phase2_learning_rate)
    return model


def get_callbacks(output_dir, phase="phase1"):
    from pathlib import Path
    out = Path(output_dir)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=4, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out / "models" / f"best_sota_{phase}.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(out / "logs" / f"sota_{phase}_log.csv"), append=False,
        ),
    ]


if __name__ == "__main__":
    cfg = SOTAConfig()
    model, base = build_efficientnet(cfg)
    model = compile_model(model)
    model.summary()
