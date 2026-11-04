"""
train.py — Hybrid MobileNetV2 Defect Classifier Training
=========================================================
Builds a frozen MobileNetV2 base with custom classification head,
trains on augmented data, validates on clean raw data, and saves
the best model checkpoint.

Usage:
    python src/train.py
    python src/train.py --epochs 30 --batch_size 32
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Rescaling
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Paths (resolved relative to this script) ──────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # src/
PROJECT_ROOT = SCRIPT_DIR.parent                  # project root

DEFAULT_AUG_DIR   = str(PROJECT_ROOT / "data" / "augmented" / "train")
DEFAULT_VAL_DIR   = str(PROJECT_ROOT / "data" / "raw" / "validation" / "images")
DEFAULT_MODEL_OUT = str(PROJECT_ROOT / "models" / "hybrid_defect_model.keras")
DEFAULT_PLOT_OUT  = str(PROJECT_ROOT / "outputs" / "training_history.png")

# Class names must match the alphabetically sorted directory names
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]
NUM_CLASSES  = len(CLASS_NAMES)
IMAGE_SIZE   = (200, 200)
IMAGE_SHAPE  = (200, 200, 3)   # RGB (grayscale images loaded as 3-channel)


# ── Data pipeline ──────────────────────────────────────────────────────────────

def build_datasets(aug_dir: str, val_dir: str, batch_size: int):
    """
    Load train (augmented) and validation (raw) datasets from directory trees.
    color_mode='rgb' ensures grayscale images are duplicated to 3 channels
    automatically by TensorFlow, matching MobileNetV2's expected input.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        aug_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",        # auto-converts grayscale -> 3-channel
        label_mode="int",        # integer labels for sparse_categorical_crossentropy
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        label_mode="int",
        shuffle=False,           # keep validation order deterministic
        seed=42,
    )

    # Confirm class order matches our expected CLASS_NAMES
    print(f"\nDetected training classes : {train_ds.class_names}")
    print(f"Detected validation classes: {val_ds.class_names}")

    # Apply pixel normalization [0, 255] -> [0.0, 1.0] via a Rescaling layer
    # applied to batches. Done in pipeline so Grad-CAM receives normalized arrays.
    rescale = Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (rescale(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch to overlap data loading with GPU computation
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ── Model construction ─────────────────────────────────────────────────────────

def build_model() -> Model:
    """
    Construct the hybrid model:
      - Frozen MobileNetV2 as feature extractor
      - Custom classification head with a named Conv2D layer ('custom_conv')
        that Grad-CAM will target for gradient extraction.
    """
    # Load MobileNetV2 without the top classification head
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SHAPE
    )
    # Freeze the entire base — we only train the custom head
    base_model.trainable = False

    # ── Custom classification head ─────────────────────────────────────────────
    x = base_model.output

    # Named Conv2D — this layer is the Grad-CAM target (see gradcam.py)
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), name="custom_conv")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Downsample before global pooling to reduce parameter count
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Collapse spatial dimensions to a feature vector
    x = GlobalAveragePooling2D()(x)

    # Regularise against overfitting on the small dataset
    x = Dropout(0.5)(x)

    # Final classification layer — softmax for multi-class probabilities
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs,
                  name="hybrid_defect_classifier")
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(aug_dir: str, val_dir: str, epochs: int, batch_size: int,
          model_out: str, plot_out: str) -> None:

    print("\n" + "="*60)
    print("  NEU-DET Hybrid Defect Classifier — Training")
    print("="*60)

    # Ensure output directories exist
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_out).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading datasets...")
    train_ds, val_ds = build_datasets(aug_dir, val_dir, batch_size)

    # ── 2. Build model ─────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model()
    model.summary()

    # ── 3. Compile ─────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── 4. Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        # Save only the best checkpoint based on validation accuracy
        ModelCheckpoint(
            filepath=model_out,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if val_accuracy doesn't improve for `patience` epochs
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # ── 5. Train ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for up to {epochs} epochs (early stopping patience=5)...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # ── 6. Final evaluation ────────────────────────────────────────────────────
    print("\n[4/4] Final evaluation on validation set...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Final Val Loss    : {val_loss:.4f}")
    print(f"  Final Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"\n  Model saved to: {model_out}")

    # ── 7. Plot training history ───────────────────────────────────────────────
    _save_history_plot(history, plot_out)
    print(f"  Training plot saved to: {plot_out}")
    print("\n" + "="*60 + "\n")


def _save_history_plot(history, plot_out: str) -> None:
    """Save a two-panel training history plot (accuracy + loss)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy panel
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss panel
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=100)
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the hybrid MobileNetV2 defect classifier"
    )
    parser.add_argument("--aug_dir",    default=DEFAULT_AUG_DIR,
                        help="Path to augmented training images (class subfolders)")
    parser.add_argument("--val_dir",    default=DEFAULT_VAL_DIR,
                        help="Path to validation images (class subfolders)")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Maximum number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--model_out",  default=DEFAULT_MODEL_OUT,
                        help="Output path for saved .keras model")
    parser.add_argument("--plot_out",   default=DEFAULT_PLOT_OUT,
                        help="Output path for training history plot")
    args = parser.parse_args()

    train(
        aug_dir=args.aug_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_out=args.model_out,
        plot_out=args.plot_out
    )
