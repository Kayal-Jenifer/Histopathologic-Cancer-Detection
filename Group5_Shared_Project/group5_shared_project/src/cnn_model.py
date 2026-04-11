from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore


@dataclass
class CNNConfig:
    input_shape: tuple = (96, 96, 3)
    num_classes: int = 1
    conv_blocks: tuple = (32, 64, 128, 256)
    dense_units: int = 256
    dropout_rate: float = 0.4
    use_batchnorm: bool = True
    l2_weight: float = 1e-4


def _conv_block(x, filters: int, use_batchnorm: bool, l2_weight: float):
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_weight),
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_weight),
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def build_custom_cnn(config: CNNConfig) -> tf.keras.Model:
    inputs = layers.Input(shape=config.input_shape, name="input_image")
    x = inputs

    for filters in config.conv_blocks:
        x = _conv_block(x, filters, config.use_batchnorm, config.l2_weight)
        x = layers.Dropout(config.dropout_rate * 0.5 if filters < 128 else config.dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(config.l2_weight),
    )(x)
    x = layers.Dropout(config.dropout_rate)(x)

    outputs = layers.Dense(
        config.num_classes,
        activation="sigmoid" if config.num_classes == 1 else "softmax",
        name="prediction",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="HistopathologicCustomCNN")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
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


if __name__ == "__main__":
    cfg = CNNConfig()
    model = build_custom_cnn(cfg)
    model = compile_model(model)
    model.summary()
