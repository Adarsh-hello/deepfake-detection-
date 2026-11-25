"""Held-out test evaluation generating Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, and Matplotlib charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf

from dataset_prep import make_tf_dataset


def evaluate_saved_model(splits_csv: Path, artifacts_dir: Path) -> None:
    model_path = artifacts_dir / "deepfake_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    df = pd.read_csv(splits_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_ds = make_tf_dataset(test_df, batch_size=32, training=False)

    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(test_ds).ravel()
    y_true = test_df["label"].to_numpy(dtype=int)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "thresholds": [round(float(x), 4) for x in thresholds],
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=["REAL", "FAKE"], output_dict=True, zero_division=0
        ),
    }

    (artifacts_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(metrics["classification_report"]).T.to_csv(
        artifacts_dir / "classification_report.csv"
    )

    # 1. Training & Validation Accuracy / Loss Charts
    hist_path = artifacts_dir / "training_history.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(hist_df["epoch"], hist_df["accuracy"], label="Train Accuracy")
        axes[0].plot(hist_df["epoch"], hist_df["val_accuracy"], label="Val Accuracy")
        axes[0].set_title("CNN Training & Validation Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(hist_df["epoch"], hist_df["loss"], label="Train Loss")
        axes[1].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
        axes[1].set_title("CNN Training & Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Binary Crossentropy Loss")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(artifacts_dir / "accuracy_loss_curves.png", dpi=150)
        plt.close(fig)

    # 2. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Test Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. ROC-AUC Curve Plot
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve (Test Split)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake_model.keras on held-out test set")
    parser.add_argument(
        "--splits-csv", type=Path, default=Path("python_pipeline/artifacts/splits.csv")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("python_pipeline/artifacts")
    )
    args = parser.parse_args()
    evaluate_saved_model(args.splits_csv, args.artifacts_dir)
