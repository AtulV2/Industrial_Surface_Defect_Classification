"""
gradcam.py — Grad-CAM Explainability for Defect Classifier
===========================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
targeting the 'custom_conv' layer of the hybrid model.

The heatmap highlights image regions that most influenced the model's
prediction, making the classification decision interpretable.

Reference: Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations
           from Deep Networks via Gradient-based Localization"

Usage (as module):
    from gradcam import compute_gradcam
    overlay, pred_class, confidence, raw_heatmap = compute_gradcam(model, img_array)

Usage (standalone test):
    python src/gradcam.py
"""

import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
GRADCAM_LAYER = "custom_conv"   # Layer whose gradients we extract

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

# ── Core Grad-CAM function ─────────────────────────────────────────────────────

def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int = None
):
    """
    Compute Grad-CAM heatmap and superimpose it on the original image.

    Args:
        model     : Loaded Keras model (expects input shape (1, 200, 200, 3), [0,1] float32)
        img_array : Preprocessed image array, shape (1, 200, 200, 3), dtype float32, values in [0, 1]
        class_idx : Target class index. If None, uses the predicted (top-1) class.

    Returns:
        heatmap_overlay  (np.ndarray uint8,  H×W×3 BGR) — colormap overlay on original image
        predicted_class  (int)                           — predicted class index
        confidence       (float)                         — softmax probability of predicted class
        raw_heatmap      (np.ndarray float32, H×W)       — normalized heatmap [0, 1], for custom use
    """
    # ── 1. Build gradient model ────────────────────────────────────────────────
    # We need two outputs from a single forward pass:
    #   a) The feature maps produced by 'custom_conv' (spatial activations)
    #   b) The final softmax predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(GRADCAM_LAYER).output,   # shape: (1, H, W, 128)
            model.output                               # shape: (1, 6)
        ]
    )

    # ── 2. Record forward pass and compute gradients ───────────────────────────
    with tf.GradientTape() as tape:
        # Cast to float32 to ensure gradient computation works correctly
        inputs = tf.cast(img_array, tf.float32)
        # tape.watch needed because inputs is not a tf.Variable
        tape.watch(inputs)

        conv_outputs, predictions = grad_model(inputs)

        # Determine which class to explain
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))

        # Extract the score for the target class (pre-softmax equivalent:
        # using the softmax output is acceptable for Grad-CAM visualisation)
        target_score = predictions[:, class_idx]

    # Gradients of the target class score w.r.t. the conv layer's output
    grads = tape.gradient(target_score, conv_outputs)   # shape: (1, H, W, 128)

    # ── 3. Pool gradients over spatial dimensions ──────────────────────────────
    # Global average pool: one scalar weight per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # shape: (128,)

    # ── 4. Weight feature maps by their pooled gradients ──────────────────────
    conv_outputs = conv_outputs[0]                          # shape: (H, W, 128)
    # Multiply each channel by its importance weight
    weighted = conv_outputs * pooled_grads                  # broadcasting: (H, W, 128)

    # Sum across channels → scalar importance map
    heatmap = tf.reduce_sum(weighted, axis=-1)              # shape: (H, W)

    # ── 5. Apply ReLU and normalize ───────────────────────────────────────────
    # ReLU: keep only features that positively contribute to the target class
    heatmap = tf.nn.relu(heatmap).numpy()                   # shape: (H, W)

    # Normalize to [0, 1] — avoid division by zero if heatmap is all zeros
    heatmap_max = heatmap.max()
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max

    raw_heatmap = heatmap.astype(np.float32)                # save before colormap

    # ── 6. Resize heatmap to match input image dimensions ─────────────────────
    h, w = img_array.shape[1], img_array.shape[2]           # (200, 200)
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # ── 7. Apply colormap and overlay on original image ───────────────────────
    # Convert normalized heatmap [0,1] → uint8 [0,255] for colormap application
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # (H, W, 3) BGR

    # Reconstruct the original image in BGR uint8 for overlay
    orig_img = img_array[0]                                 # (200, 200, 3), [0,1]
    orig_bgr = cv2.cvtColor(
        np.uint8(orig_img * 255), cv2.COLOR_RGB2BGR
    )

    # Weighted blend: 60% original + 40% heatmap colormap
    heatmap_overlay = cv2.addWeighted(orig_bgr, 0.6, colormap, 0.4, 0)

    # ── 8. Extract prediction info ─────────────────────────────────────────────
    confidence = float(predictions[0][class_idx])

    return heatmap_overlay, class_idx, confidence, raw_heatmap


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent

    model_path  = project_root / "models" / "hybrid_defect_model.keras"
    val_img_dir = project_root / "data" / "raw" / "validation" / "images"
    output_dir  = project_root / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        print("        Please run src/train.py first.")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # ── Pick a sample image from the validation set ────────────────────────────
    sample_img_path = None
    for class_dir in sorted(val_img_dir.iterdir()):
        if class_dir.is_dir():
            imgs = list(class_dir.glob("*.bmp")) + \
                   list(class_dir.glob("*.jpg")) + \
                   list(class_dir.glob("*.png"))
            if imgs:
                sample_img_path = imgs[0]
                sample_class    = class_dir.name
                break

    if sample_img_path is None:
        print("[ERROR] No validation images found.")
        sys.exit(1)

    print(f"Sample image: {sample_img_path.name}  (true class: {sample_class})")

    # ── Preprocess image ───────────────────────────────────────────────────────
    img_bgr  = cv2.imread(str(sample_img_path))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (200, 200))
    img_array   = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # ── Run Grad-CAM ───────────────────────────────────────────────────────────
    overlay, pred_class, confidence, _ = compute_gradcam(model, img_array)

    pred_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
    print(f"Predicted class: {pred_name}  ({confidence*100:.1f}% confidence)")

    # ── Save side-by-side output ───────────────────────────────────────────────
    # Build a side-by-side composite: original (grayscale→BGR) | Grad-CAM overlay
    orig_bgr = cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR)
    combined = np.hstack([orig_bgr, overlay])

    out_path = output_dir / f"gradcam_sample_{sample_class}.bmp"
    cv2.imwrite(str(out_path), combined)
    print(f"Grad-CAM overlay saved to: {out_path}")

