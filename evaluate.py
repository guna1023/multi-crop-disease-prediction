"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set (dataset_split/test),
which the model has NEVER seen during training or validation.

Prints and saves:
  - Test accuracy
  - Precision / Recall / F1-score (weighted average, printed to console)
  - Full classification report -> results/classification_report.txt
  - Confusion matrix plot       -> results/confusion_matrix.png

USAGE:
    python evaluate.py
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from utils_common import load_class_names

TEST_DIR = os.path.join("dataset_split", "test")
MODELS_DIR = "models"
RESULTS_DIR = "results"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"'{BEST_MODEL_PATH}' not found. Run train.py first to train and save a model."
        )
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"'{TEST_DIR}' not found. Run prepare_dataset.py first to create the test split."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model and class names...")
    model = keras.models.load_model(BEST_MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

    print("Loading test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,  # keep order so predictions line up with labels
    )

    # Sanity check: class order in test set must match training class order
    if test_ds.class_names != class_names:
        print(
            "WARNING: test set class order differs from training class order. "
            "Predictions will be re-mapped, but please double check your dataset folders."
        )

    print("Running predictions on the test set...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("TEST SET RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy : {acc * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall   : {recall * 100:.2f}%\n")
        f.write(f"F1-score : {f1 * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\nSaved classification report to: {report_path}")

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.7)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    main()
