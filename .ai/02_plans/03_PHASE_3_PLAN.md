# Phase 3 Plan — Grad-CAM Explainability
**Auto-approved (autopilot mode):** 2026-11-04

## Goal
Implement a reusable Grad-CAM module targeting the `custom_conv` layer.

## Function signature
```python
def compute_gradcam(model, img_array, class_idx=None)
    -> (heatmap_overlay: ndarray uint8 BGR, predicted_class: int, confidence: float, raw_heatmap: ndarray float32)
```

## Algorithm
1. Gradient model: outputs [custom_conv activations, final predictions]
2. GradientTape → gradients of target class score w.r.t. custom_conv
3. Pool gradients → weight feature maps → ReLU → normalize → resize to 200×200
4. Apply COLORMAP_JET → addWeighted overlay (0.6 original + 0.4 heatmap)

## Deliverable
`src/gradcam.py` — importable module + standalone test block

