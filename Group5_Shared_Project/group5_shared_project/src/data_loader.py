from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE


def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["filename"] = df["id"].astype(str) + ".tif"
    return df


def split_labels(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.10, random_state: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _parse_image(filename, label, image_dir, image_size=(96, 96), augment=False):
    image_path = tf.strings.join([image_dir, "/", filename])
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.10)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(df, image_dir: str, batch_size: int = 64, shuffle: bool = False, augment: bool = False):
    image_dir = str(Path(image_dir))
    ds = tf.data.Dataset.from_tensor_slices((df["filename"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda x, y: _parse_image(x, y, image_dir=image_dir, augment=augment),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
