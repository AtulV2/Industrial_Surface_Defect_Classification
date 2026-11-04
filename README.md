# Industrial Surface Defect Classification

A deep learning system that classifies steel surface defects from the **NEU Surface Defect Database** and explains its predictions using **Grad-CAM** visualizations.

## Classes
| Class | Description |
|---|---|
| `crazing` | Network of fine cracks on the surface |
| `inclusion` | Foreign material embedded in the steel |
| `patches` | Surface patches or blotches |
| `pitted_surface` | Small pits or holes on the surface |
| `rolled-in_scale` | Scale rolled into the surface during processing |
| `scratches` | Linear surface scratches |

## Architecture
- **Base:** MobileNetV2 (ImageNet pre-trained, frozen)
- **Custom Top:** Conv2D → BatchNorm → ReLU → MaxPool → GAP → Dropout → Dense(6)
- **Explainability:** Grad-CAM targeting the `custom_conv` layer

## Project Structure
```
├── data/
│   ├── raw/              # Original NEU-DET dataset (train + validation)
│   └── augmented/        # Offline-augmented training images (originals + variants)
├── src/
│   ├── augment.py        # Offline augmentation script
│   ├── train.py          # Model training script
│   ├── gradcam.py        # Grad-CAM explainability module
│   └── app.py            # Streamlit web application
├── models/               # Saved .keras model files
└── outputs/              # Grad-CAM overlays and training plots
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate augmented training data
```bash
python src/augment.py
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Launch the web app
```bash
streamlit run src/app.py
```

## Dataset
The [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) contains 1,800 grayscale images (200×200) across 6 defect classes (300 images per class).

Place the dataset under `data/raw/` with the following structure:
```
data/raw/
├── train/images/{class_name}/
└── validation/images/{class_name}/
```

