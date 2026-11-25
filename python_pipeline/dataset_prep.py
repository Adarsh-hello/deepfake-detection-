"""Dataset preparation, 70/15/15 stratified splitting, 128x128 preprocessing, and tf.data pipelines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMAGE_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE


def scan_dataset(real_dir: Path, fake_dir: Path) -> pd.DataFrame:
    """Scan Real/Fake directories, verify image integrity, and deduplicate by SHA-256."""
    seen_hashes: set[str] = set()
    rows: List[Dict] = []

    for label, directory in [(0, real_dir), (1, fake_dir)]:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue

            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            rows.append(
                {
                    "filepath": str(path.resolve()),
                    "label": label,
                    "class_name": "FAKE" if label == 1 else "REAL",
                    "sha256": sha,
                }
            )

    if len(rows) < 10:
        raise RuntimeError(f"Insufficient valid images found ({len(rows)}). Need at least 10.")

    return pd.DataFrame(rows)


def create_stratified_splits(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset strictly into 70% train, 15% validation, and 15% held-out test."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return train_df, val_df, test_df


def preprocess_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode, resize to 128x128 RGB, and normalize to [0, 1] identically to training."""
    img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def _load_and_preprocess(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)


def build_augmentation_layer() -> tf.keras.Sequential:
    """Data augmentation applied strictly to the training split."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )


def make_tf_dataset(
    df: pd.DataFrame, batch_size: int = 32, training: bool = False
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values)
    )
    if training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        aug = build_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 70/15/15 Real vs Fake dataset manifest")
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("python_pipeline/artifacts/splits.csv")
    )
    args = parser.parse_args()

    full_df = scan_dataset(args.real_dir, args.fake_dir)
    tr, va, te = create_stratified_splits(full_df)
    combined = pd.concat([tr, va, te], ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)

    summary = {
        "total_valid": int(len(combined)),
        "train": int(len(tr)),
        "val": int(len(va)),
        "test": int(len(te)),
    }
    print(json.dumps(summary, indent=2))
