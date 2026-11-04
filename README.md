# 🔬 Industrial Surface Defect Classification

> An AI-powered system that automatically identifies steel surface defects and **visually explains** its decisions using Grad-CAM heatmaps.

---

## 🎯 What Does This Project Do?

You upload a photo of a steel surface → the system tells you **what defect it is** and **highlights exactly where** on the image the AI looked to make that decision.

```
 📷 Upload Image  →  🤖 AI Classifies  →  🌡️ Heatmap Shows Why
   (steel surface)     (e.g. "crazing")     (red = high attention)
```

**Validation Accuracy: 99.72%** across 6 defect types.

---

## 🏭 The 6 Defect Classes

| # | Class | Description | Appearance |
|---|---|---|---|
| 1 | **Crazing** | Network of fine cracks | Spider-web pattern across surface |
| 2 | **Inclusion** | Foreign material embedded in steel | Dark spots or patches of contamination |
| 3 | **Patches** | Surface blotches | Irregular light/dark patches |
| 4 | **Pitted Surface** | Small pits or holes | Rough, pock-marked texture |
| 5 | **Rolled-in Scale** | Oxide scale pressed into surface | Elongated dark marks along rolling direction |
| 6 | **Scratches** | Linear surface scratches | Fine parallel lines or gouges |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app (from project root)
streamlit run src/app.py
```

Then open **http://localhost:8501** in your browser and upload any surface image.

---

## 🗺️ Full System Flow

```mermaid
flowchart TD
    A["📁 NEU-DET Dataset\n1,800 grayscale images\n6 classes × 300 images"] --> B["🔀 Train/Validation Split\n80% train · 20% validation\n(pre-split in dataset)"]

    B --> C["📂 data/raw/train/images/\n240 images per class"]
    B --> D["📂 data/raw/validation/images/\n60 images per class"]

    C --> E["⚙️ Offline Augmentation\nsrc/augment.py\nRotate · Flip · Brightness · Translate\n× 4 copies per image"]

    E --> F["📂 data/augmented/train/\n1,200 images per class\n7,200 total"]

    F --> G["🧠 Model Training\nsrc/train.py\nMobileNetV2 + Custom Head"]
    D --> G

    G --> H["💾 models/hybrid_defect_model.keras\nBest checkpoint — Val Acc: 99.72%"]

    H --> I["🌐 Streamlit App\nsrc/app.py\nstreamlit run src/app.py"]
    I --> J["📤 Upload Image"]
    J --> K["🔍 Predict Class + Confidence"]
    J --> L["🌡️ Grad-CAM Heatmap\nsrc/gradcam.py"]
    K --> M["📊 Results Dashboard\nClass · Confidence · Bar Chart · Overlay"]
    L --> M
```

---

## 🧠 Model Architecture

```mermaid
flowchart TD
    IN["Input Image\n200×200×3 RGB"] --> SCALE["Rescaling ÷255\nNormalize to 0–1"]
    SCALE --> BASE["MobileNetV2 Base\nImageNet pretrained\n🔒 Frozen — not trained\nOutput: 7×7×1280 feature maps"]
    BASE --> CC["Conv2D — 128 filters 3×3\nname: 'custom_conv'\n⬅️ Grad-CAM targets this layer"]
    CC --> BN["Batch Normalization\nStabilizes training"]
    BN --> RELU["ReLU Activation\nRemoves negative values"]
    RELU --> MP["MaxPooling 2×2\nReduces spatial size"]
    MP --> GAP["Global Average Pooling\nCollapses spatial dims → vector"]
    GAP --> DO["Dropout 50%\nPrevents overfitting"]
    DO --> OUT["Dense — 6 neurons\nSoftmax activation\n→ Probabilities per class"]
```

---

## 🌡️ How Grad-CAM Works

```mermaid
flowchart LR
    A["Input Image\n200×200"] --> B["Forward Pass\nthrough model"]
    B --> C["custom_conv\nFeature Maps\n7×7×128"]
    B --> D["Prediction\ne.g. crazing=0.99"]
    D --> E["Backprop Gradients\nthrough custom_conv"]
    E --> F["Pool Gradients\nper channel weight"]
    F --> G["Weight Feature Maps\nby importance"]
    C --> G
    G --> H["ReLU + Normalize\nheatmap 7×7"]
    H --> I["Resize to 200×200\ncv2.resize"]
    I --> J["Apply JET Colormap\nblue=low · red=high"]
    J --> K["Blend with Original\n60% image + 40% heatmap"]
    K --> L["🌡️ Heatmap Overlay\nshows WHERE model looked"]
```

---

## 📂 Project Structure

```
Industrial_Surface_Defect_Classification/
│
├── 📁 data/
│   ├── raw/                        ← Original NEU-DET dataset (never modified)
│   │   ├── train/images/{class}/   ← 240 images × 6 classes
│   │   └── validation/images/{class}/  ← 60 images × 6 classes
│   └── augmented/
│       └── train/{class}/          ← 1,200 images × 6 classes (orig + aug)
│
├── 📁 src/
│   ├── augment.py      ← Step 1: Generate augmented training images
│   ├── train.py        ← Step 2: Build + train the model
│   ├── gradcam.py      ← Step 3: Grad-CAM explainability module
│   └── app.py          ← Step 4: Streamlit web interface
│
├── 📁 models/
│   └── hybrid_defect_model.keras   ← Saved trained model
│
├── 📁 outputs/
│   ├── training_history.png        ← Accuracy/loss curves
│   └── gradcam_sample_*.bmp        ← Sample heatmap overlays
│
├── 📁 docs/                        ← Full documentation
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_DATA_PIPELINE.md
│   ├── 03_MODEL_ARCHITECTURE.md
│   ├── 04_GRADCAM_EXPLAINED.md
│   ├── 05_INFERENCE_FLOW.md
│   └── 06_METRICS_AND_RESULTS.md
│
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

---

## 📈 Training Results

| Metric | Value |
|---|---|
| Training images | 7,200 (augmented) |
| Validation images | 360 (clean originals) |
| Best epoch | 6 of 30 |
| **Validation accuracy** | **99.72%** |
| Validation loss | 0.0098 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet) |
| Explainability | Grad-CAM (tf.GradientTape) |
| Augmentation | OpenCV + NumPy |
| Web UI | Streamlit |
| Language | Python 3.10+ |

---

## 📖 Documentation

| Document | Contents |
|---|---|
| [Project Overview](docs/01_PROJECT_OVERVIEW.md) | Problem statement, goals, solution approach |
| [Data Pipeline](docs/02_DATA_PIPELINE.md) | Dataset details, augmentation strategy, data flow |
| [Model Architecture](docs/03_MODEL_ARCHITECTURE.md) | Layer-by-layer breakdown, design decisions |
| [Grad-CAM Explained](docs/04_GRADCAM_EXPLAINED.md) | How explainability works, step by step |
| [Inference Flow](docs/05_INFERENCE_FLOW.md) | Upload → predict → display, end-to-end |
| [Metrics & Verification](docs/06_METRICS_AND_RESULTS.md) | Training history, evaluation curves, reliability |
