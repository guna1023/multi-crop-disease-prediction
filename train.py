"""
train.py
--------
Trains a MobileNetV2 transfer-learning model to classify leaf images into
disease classes for Tomato and Corn.

BEFORE RUNNING THIS:
  1. Run prepare_dataset.py first. It creates dataset_split/train,
     dataset_split/val, dataset_split/test from dataset_raw/.

WHAT THIS SCRIPT DOES:
  1. Loads dataset_split/train and dataset_split/val.
  2. Automatically detects class names from folder names (no hard-coding).
  3. Builds a MobileNetV2 model with data-augmentation and preprocessing
     baked directly into the model, so predict.py / app.py can feed raw
     images without re-implementing preprocessing.
  4. Stage 1: trains only the classification head (base frozen).
  5. Stage 2: fine-tunes the top layers of MobileNetV2 with a tiny
     learning rate.
  6. Saves the best model (by validation accuracy) to models/best_model.keras
  7. Saves class names to models/class_names.json
  8. Saves a training curve plot to results/training_history.png

RESUME SUPPORT:
  Every epoch, the current full model is also saved to
  models/last_checkpoint.keras (regardless of whether it was the best).
  If training is interrupted (Ctrl+C, power cut, laptop sleep, etc.),
  just run:
        python train.py --resume
  and it will continue from models/last_checkpoint.keras instead of
  starting from ImageNet weights again.

USAGE:
    python train.py
    python train.py --epochs 15 --fine_tune_epochs 10
    python train.py --resume
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_DIR = os.path.join("dataset_split", "train")
VAL_DIR = os.path.join("dataset_split", "val")

MODELS_DIR = "models"
RESULTS_DIR = "results"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
LAST_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "last_checkpoint.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
HISTORY_PLOT_PATH = os.path.join(RESULTS_DIR, "training_history.png")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

DEFAULT_STAGE1_EPOCHS = 15
DEFAULT_STAGE2_EPOCHS = 10
FINE_TUNE_UNFREEZE_LAYERS = 30  # how many top layers of MobileNetV2 to unfreeze


def build_datasets():
    """Load train/val datasets and return them along with class names."""
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names  # auto-detected from folder names
    print(f"\nDetected {len(class_names)} classes:")
    for name in class_names:
        print(f"  - {name}")

    # Print image counts per class (from the raw train folder)
    print("\nImages per class (train split):")
    for name in class_names:
        folder = os.path.join(TRAIN_DIR, name)
        count = len(os.listdir(folder))
        print(f"  {name:40s}: {count}")

    # Performance: cache + prefetch (data stays local, no internet needed)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_perf = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds_perf = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, train_ds_perf, val_ds_perf, class_names


def compute_class_weights(class_names):
    """Compute class weights to handle imbalanced datasets."""
    labels = []
    for idx, name in enumerate(class_names):
        folder = os.path.join(TRAIN_DIR, name)
        count = len([f for f in os.listdir(folder)])
        labels.extend([idx] * count)

    labels = np.array(labels)
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=labels
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print("\nComputed class weights (to handle imbalance):")
    for k, v in class_weight_dict.items():
        print(f"  class {k} ({class_names[k]}): {v:.3f}")
    return class_weight_dict


def build_model(num_classes):
    """
    Build a MobileNetV2 transfer-learning model.
    Data augmentation + preprocessing are included INSIDE the model so that
    predict.py and app.py can simply feed a raw 0-255 RGB image without
    re-implementing any preprocessing logic themselves.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Stage 1: frozen base

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)  # active only when training=True
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def get_callbacks(monitor="val_accuracy", mode="max"):
    return [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            LAST_CHECKPOINT_PATH,
            save_best_only=False,  # always overwrite -> resume point
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_history(history_stage1, history_stage2=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    acc = history_stage1.history["accuracy"]
    val_acc = history_stage1.history["val_accuracy"]
    loss = history_stage1.history["loss"]
    val_loss = history_stage1.history["val_loss"]

    if history_stage2 is not None:
        acc += history_stage2.history["accuracy"]
        val_acc += history_stage2.history["val_accuracy"]
        loss += history_stage2.history["loss"]
        val_loss += history_stage2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    if history_stage2 is not None:
        stage1_len = len(history_stage1.history["accuracy"])
        plt.axvline(x=stage1_len, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.legend(loc="lower right")
    plt.title("Training / Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    if history_stage2 is not None:
        stage1_len = len(history_stage1.history["loss"])
        plt.axvline(x=stage1_len, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.legend(loc="upper right")
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"\nSaved training history plot to: {HISTORY_PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train multi-crop disease detector")
    parser.add_argument("--epochs", type=int, default=DEFAULT_STAGE1_EPOCHS,
                         help="Epochs for stage 1 (frozen base)")
    parser.add_argument("--fine_tune_epochs", type=int, default=DEFAULT_STAGE2_EPOCHS,
                         help="Epochs for stage 2 (fine-tuning)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume training from models/last_checkpoint.keras")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading datasets...")
    _, train_ds, val_ds, class_names = build_datasets()

    # Save class names immediately so evaluate.py / predict.py can use them
    # even if training is interrupted later.
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to: {CLASS_NAMES_PATH}")

    class_weight_dict = compute_class_weights(class_names)

    if args.resume and os.path.exists(LAST_CHECKPOINT_PATH):
        print(f"\nResuming from checkpoint: {LAST_CHECKPOINT_PATH}")
        model = keras.models.load_model(LAST_CHECKPOINT_PATH)
        # Find the base MobileNetV2 layer inside the loaded model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower():
                base_model = layer
                break
    else:
        if args.resume:
            print(
                "\n--resume was set but no checkpoint found at "
                f"'{LAST_CHECKPOINT_PATH}'. Starting fresh instead."
            )
        model = None
        base_model = None

    if model is None:
        model, base_model = build_model(num_classes=len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.summary()

    # -----------------------------------------------------------------
    # STAGE 1: train classification head with base frozen
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: Training classification head (MobileNetV2 base frozen)")
    print("=" * 70)

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(),
    )

    # -----------------------------------------------------------------
    # STAGE 2: fine-tune top layers of MobileNetV2 with a tiny LR
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-tuning top layers of MobileNetV2")
    print("=" * 70)

    if base_model is not None:
        base_model.trainable = True
        # Freeze all layers except the last N
        for layer in base_model.layers[:-FINE_TUNE_UNFREEZE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight_dict,
            callbacks=get_callbacks(),
        )
    else:
        print("Could not locate base MobileNetV2 layer for fine-tuning. Skipping stage 2.")
        history_stage2 = None

    plot_history(history_stage1, history_stage2)

    print("\nTraining complete!")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print(f"Class names saved at: {CLASS_NAMES_PATH}")
    print(f"Training curves saved at: {HISTORY_PLOT_PATH}")
    print("\nNext step: run 'python evaluate.py' to check performance on the test set.")


if __name__ == "__main__":
    main()
