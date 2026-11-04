# AUDIT_002 — Phase 1: Offline Augmentation
**Date:** 2026-11-04  
**Phase:** 1 — Offline Augmentation  
**Status:** ✅ COMPLETE

## Files Added
| File | Purpose |
|---|---|
| `src/augment.py` | Offline augmentation script (OpenCV + NumPy, seeded) |

## Execution Output
```
Source : data/raw/train/images
Output : data/augmented/train
Copies : 4 augmented variants per original

[             crazing]  originals:  240  total written:  1200
[           inclusion]  originals:  240  total written:  1200
[             patches]  originals:  240  total written:  1200
[      pitted_surface]  originals:  240  total written:  1200
[     rolled-in_scale]  originals:  240  total written:  1200
[           scratches]  originals:  240  total written:  1200

Augmentation complete — 7200 images written
```

## Phase 1 Requirements Check
- [x] Reads from data/raw/train/images/{class}/
- [x] Writes originals + 4 augmented copies to data/augmented/train/{class}/
- [x] 240 originals x 5 (1 original copy + 4 aug) = 1200 per class x 6 = 7200 total
- [x] Seed=42 set for reproducibility
- [x] Augmentations: rotation +/-15, horizontal flip 50%, brightness +/-20%, translation +/-10%
- [x] data/raw/ untouched — ground truth preserved
- [x] Validation data NOT augmented

## Dependencies Added
- opencv-python — used for image I/O and transforms

## Known Issues / Risks
- 240 originals per class (not 300 as estimated) — dataset has 80/20 split applied upstream. Total is correct and sufficient.
