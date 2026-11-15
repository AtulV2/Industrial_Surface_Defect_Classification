"""
app.py — Streamlit Web UI for Industrial Surface Defect Classifier
==================================================================
Provides an interactive web interface to:
  1. Upload a surface defect image
  2. Display predicted defect class + confidence score
  3. Show side-by-side: original grayscale image | Grad-CAM heatmap overlay
  4. Display a confidence bar chart across all 6 classes

Run from the PROJECT ROOT directory:
    streamlit run src/app.py

The model path is resolved relative to the working directory (project root),
so the command must be run from there.
"""

import sys
import os
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Import gradcam from the same src/ directory ────────────────────────────────
# Since app.py and gradcam.py are both in src/, we add src/ to sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gradcam import compute_gradcam   # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = _SRC_DIR.parent

# Robust model path resolution: works whether run from root or from src/
_CANDIDATE_PATHS = [
    _PROJECT_ROOT / "models" / "hybrid_defect_model.keras",
    Path("models") / "hybrid_defect_model.keras",
    Path("../models") / "hybrid_defect_model.keras",
    _SRC_DIR / "models" / "hybrid_defect_model.keras",
]

def get_model_path() -> Path:
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return p.resolve()
    return (_PROJECT_ROOT / "models" / "hybrid_defect_model.keras").resolve()

MODEL_PATH = get_model_path()

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_DESCRIPTIONS = {
    "crazing":        "Network of fine surface cracks",
    "inclusion":      "Foreign material embedded in steel",
    "patches":        "Surface blotches or patches",
    "pitted_surface": "Small pits or holes on the surface",
    "rolled-in_scale":"Scale pressed into the surface during rolling",
    "scratches":      "Linear surface scratches",
}

IMAGE_SIZE = (200, 200)


# ── Model loading (cached — loaded only once per Streamlit session) ────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model() -> tf.keras.Model:
    """Load and cache the trained Keras model."""
    resolved_path = get_model_path()
    if not resolved_path.exists():
        st.error(
            f"Model file not found at `{resolved_path}`.\n\n"
            f"Searched locations:\n" + "\n".join(f"- `{p}`" for p in _CANDIDATE_PATHS) + "\n\n"
            "Please run `python src/train.py` first to train and save the model."
        )
        st.stop()
    return tf.keras.models.load_model(str(resolved_path))


# ── Image preprocessing ────────────────────────────────────────────────────────

def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Convert an uploaded file to a preprocessed numpy array.

    Returns:
        img_array: shape (1, 200, 200, 3), float32, values in [0, 1]
    """
    # PIL handles all common formats (BMP, JPG, PNG, TIFF …)
    pil_img = Image.open(uploaded_file).convert("RGB")
    pil_img = pil_img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_np  = np.array(pil_img, dtype=np.float32) / 255.0   # normalize to [0,1]
    return np.expand_dims(img_np, axis=0)                     # add batch dimension


# ── Streamlit App ──────────────────────────────────────────────────────────────

def main():
    # ── Page config ────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="NEU Defect Classifier",
        page_icon="🔬",
        layout="wide"
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown(
            "**Industrial Surface Defect Classification**\n\n"
            "Model: MobileNetV2 (frozen) + Custom Head\n\n"
            "Dataset: NEU Surface Defect Database\n\n"
            "Explainability: Grad-CAM on `custom_conv` layer"
        )
        st.divider()
        st.subheader("📋 Defect Classes")
        for name, desc in CLASS_DESCRIPTIONS.items():
            st.markdown(f"**{name}**  \n_{desc}_")

    # ── Main title ─────────────────────────────────────────────────────────────
    st.title("🔬 Industrial Surface Defect Classifier")
    st.markdown(
        "Upload a steel surface image to classify the defect type "
        "and visualise the model's attention via **Grad-CAM**."
    )
    st.divider()

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Accepts BMP, JPG, PNG images. For best results use 200×200 grayscale images."
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started.")
        return

    # ── Preprocess ─────────────────────────────────────────────────────────────
    img_array = preprocess_image(uploaded_file)

    # ── Predict ────────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        overlay, pred_class, confidence, raw_heatmap = compute_gradcam(
            model, img_array
        )

    pred_name = CLASS_NAMES[pred_class]

    # ── Result header ──────────────────────────────────────────────────────────
    st.success(
        f"**Predicted Defect: {pred_name.upper()}**  —  "
        f"Confidence: **{confidence:.1%}**"
    )

    st.divider()

    # ── Side-by-side visual ────────────────────────────────────────────────────
    col_orig, col_cam = st.columns(2)

    with col_orig:
        st.subheader("📷 Original Image")
        # Display as grayscale (convert RGB → L for a true grey look)
        pil_orig = Image.open(uploaded_file).convert("L").resize(IMAGE_SIZE)
        st.image(pil_orig, caption="Uploaded Image (Grayscale)", use_container_width=True)

    with col_cam:
        st.subheader("🌡️ Grad-CAM Heatmap")
        # overlay is BGR (from OpenCV) — convert to RGB for Streamlit
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, caption="Grad-CAM Overlay (activation regions)",
                 use_container_width=True)

    st.divider()

    # ── Confidence bar chart across all classes ────────────────────────────────
    st.subheader("📊 Confidence Scores — All Classes")

    # Run full prediction to get all class probabilities
    preds = model.predict(img_array, verbose=0)[0]           # shape: (6,)
    score_dict = {name: float(score) for name, score in zip(CLASS_NAMES, preds)}

    # Streamlit bar_chart expects a dict or DataFrame
    import pandas as pd
    chart_df = pd.DataFrame.from_dict(
        score_dict, orient="index", columns=["Confidence"]
    )
    st.bar_chart(chart_df, height=300)

    # Raw scores table (expandable)
    with st.expander("Show raw scores"):
        for name, score in sorted(score_dict.items(), key=lambda x: -x[1]):
            st.write(f"**{name}**: {score:.4f} ({score*100:.2f}%)")


if __name__ == "__main__":
    main()

