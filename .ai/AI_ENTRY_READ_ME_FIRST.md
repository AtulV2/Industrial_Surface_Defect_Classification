# AI Entry Point — Read This First

## Project
**Industrial Surface Defect Classification**  
Deep learning system using MobileNetV2 + Grad-CAM on the NEU Surface Defect Database.

## Optimized Read Order (Rule 13)
1. This file
2. `.ai/01_rules/` — all 3 rule files
3. `CURRENT_STATE.md` — current phase and status
4. Most recent file in `.ai/03_checkpoints/`
5. `.ai/02_plans/PENDING_NEXT_PHASE_DRAFT.md` (if exists)

## DO NOT read historical audits, chat logs, or old checkpoints unless explicitly commanded.

## Stack
- Python, TensorFlow/Keras, Streamlit
- Dataset: NEU-DET (pre-split, 6 classes, 200×200 grayscale)
- Output files: `train.py`, `gradcam.py`, `app.py`, `requirements.txt`

