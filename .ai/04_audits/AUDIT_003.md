# AUDIT_003 — Phase 2: Model Training
**Date:** 2026-11-04  
**Phase:** 2 — Hybrid Model Training  
**Status:** COMPLETE

## Files Added
| File | Purpose |
|---|---|
| `src/train.py` | Full training pipeline — data loading, model build, train, save |
| `models/hybrid_defect_model.keras` | Best model checkpoint (saved at epoch 6) |
| `outputs/training_history.png` | Accuracy + loss curves plot |

## Execution Output
```
Found 7200 files belonging to 6 classes.  (augmented train)
Found  360 files belonging to 6 classes.  (raw validation)

Epoch  6: val_accuracy = 0.9972 — BEST — model saved
Epoch 11: early stopping (patience=5), restoring epoch 6 weights

Final Val Loss    : 0.0098
Final Val Accuracy: 0.9972 (99.72%)
```

## Phase 2 Requirements Check
- [x] image_dataset_from_directory for train + val
- [x] color_mode='rgb' — grayscale auto-converted to 3-channel
- [x] Rescaling(1/255) in pipeline
- [x] MobileNetV2 frozen base (imagenet weights, include_top=False)
- [x] Conv2D(128, name='custom_conv') > BN > ReLU > MaxPool > GAP > Dropout(0.5) > Dense(6,softmax)
- [x] Adam(1e-3) + sparse_categorical_crossentropy
- [x] ModelCheckpoint + EarlyStopping(patience=5)
- [x] models/hybrid_defect_model.keras saved
- [x] outputs/training_history.png saved

## Result
99.72% validation accuracy on 360 clean original images.
