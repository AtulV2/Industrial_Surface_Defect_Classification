# Master Roadmap — Industrial Surface Defect Classification
**Approved:** 2026-11-04  
**Status:** LOCKED — DO NOT EDIT

---

## Stack
- Python 3.10+, TensorFlow/Keras, Streamlit, OpenCV
- Dataset: NEU Surface Defect Database (NEU-DET), 6 classes, 200×200 grayscale

## Augmentation Strategy — Option B
- `data/augmented/train/{class}/` = originals + 4 augmented copies per image → single unified training source
- `data/raw/validation/images/{class}/` = untouched originals → clean evaluation only
- `data/raw/` is NEVER modified — always the ground truth reference

## Augmentation Parameters
| Transform | Range |
|---|---|
| Random Rotation | ±15° |
| Random Horizontal Flip | 50% probability |
| Random Brightness | ±20% |
| Random Translation | ±10% x/y |
| Copies per image | 4 (1440 raw → ~7200 total training images) |

---

## Full Folder Structure
```
Industrial_Surface_Defect_Classification/
├── .ai/                          ✅ done
├── data/
│   ├── raw/                      ← renamed from Data_Set_NEU-DET
│   │   ├── train/images/{6 classes}/
│   │   └── validation/images/{6 classes}/
│   └── augmented/
│       └── train/{6 classes}/    ← originals + augmented copies (Option B)
├── src/
│   ├── augment.py                ← Phase 1
│   ├── train.py                  ← Phase 2
│   ├── gradcam.py                ← Phase 3
│   └── app.py                    ← Phase 4
├── models/                       ← hybrid_defect_model.keras
├── outputs/                      ← Grad-CAM overlays, training plots
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
└── .gitignore
```

---

## Phase Roadmap
| # | Phase | Key Deliverables | Gate Condition |
|---|---|---|---|
| **0** | Repo Setup | Rename dataset → data/raw/, create all folders, git init, .gitignore, requirements.txt, README.md, CONTRIBUTING.md | git status clean, structure verified |
| **1** | Augmentation | src/augment.py — reads data/raw/train/images/, writes originals + 4 aug copies to data/augmented/train/, seeded | All 6 class folders populated, counts printed |
| **2** | Model Training | src/train.py — loads augmented/train + raw/validation, builds MobileNetV2 hybrid, trains, saves models/ | .keras file saved, val accuracy logged |
| **3** | Grad-CAM | src/gradcam.py — reusable function, gradients from custom_conv, overlay returned | Sample overlay saved to outputs/ |
| **4** | Streamlit UI | src/app.py — upload → predict → side-by-side Grad-CAM | App runs locally, full flow works |
