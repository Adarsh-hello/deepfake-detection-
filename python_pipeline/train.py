"""CNN model training script with EarlyStopping and ModelCheckpoint saving deepfake_model.keras."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from dataset_prep import make_tf_dataset
from models import build_cnn_model


def train_cnn(
    splits_csv: Path,
    artifacts_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Path:
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Test set is strictly excluded from training and validation
    train_ds = make_tf_dataset(train_df, batch_size=batch_size, training=True)
    val_ds = make_tf_dataset(val_df, batch_size=batch_size, training=False)

    # Inverse-frequency class weights for imbalance resilience
    n_real = max(1, int((train_df["label"] == 0).sum()))
    n_fake = max(1, int((train_df["label"] == 1).sum()))
    total = n_real + n_fake
    class_weight = {0: total / (2.0 * n_real), 1: total / (2.0 * n_fake)}

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "deepfake_model.keras"

    model = build_cnn_model(input_shape=(128, 128, 3), learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_df.to_csv(artifacts_dir / "training_history.csv", index=False)
    print(f"Saved best CNN checkpoint to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom Deepfake Detection CNN")
    parser.add_argument(
        "--splits-csv", type=Path, default=Path("python_pipeline/artifacts/splits.csv")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("python_pipeline/artifacts")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_cnn(args.splits_csv, args.artifacts_dir, args.epochs, args.batch_size)
