# AUDIT_002 — Phase 1: Offline Data Augmentation
**Date:** 2026-11-04  
**Phase:** 1 — Augmentation  
**Status:** ✅ COMPLETE  

---

## 1. Summary of Changes
Implemented reproducible offline data augmentation script in `src/augment.py` using OpenCV (`cv2`) and NumPy. Augmented the training dataset by generating 4 stochastic variants for each raw image while preserving the original.

## 2. Files Added / Modified
| File | Action | Purpose |
|---|---|---|
| `src/augment.py` | Added | Augmentation script with OpenCV and NumPy transforms |
| `.ai/02_plans/01_PHASE_1_PLAN.md` | Added | Approved execution plan for Phase 1 |
| `.ai/04_audits/AUDIT_002.md` | Added | This audit report |

## 3. Dependencies Added to requirements.txt
- OpenCV (`opencv-python>=4.8.0`) and NumPy (`numpy>=1.24.0`) were already declared in `requirements.txt`.
- No new external dependencies required.

## 4. Execution Output & Verification
Command executed:
```powershell
python src/augment.py
```

### Verification Counts:
- Source dataset: `data/raw/train/images/` (1,440 images total, 240 per class)
- Target dataset: `data/augmented/train/` (7,200 images total, 1,200 per class)

| Defect Class | Original Count | Augmented Copies | Total Output Count |
|---|---|---|---|
| `crazing` | 240 | 960 | 1,200 |
| `inclusion` | 240 | 960 | 1,200 |
| `patches` | 240 | 960 | 1,200 |
| `pitted_surface` | 240 | 960 | 1,200 |
| `rolled-in_scale` | 240 | 960 | 1,200 |
| `scratches` | 240 | 960 | 1,200 |
| **Total** | **1,440** | **5,760** | **7,200** |

## 5. Phase Requirements Checklist
- [x] Script created at `src/augment.py`
- [x] Uses OpenCV and NumPy only (no TensorFlow in augment script)
- [x] Reproducibility ensured with `np.random.seed(42)`
- [x] Implements all required helper functions:
  - [x] `rotate_image(img, angle_range=15)`
  - [x] `flip_image(img, prob=0.5)`
  - [x] `adjust_brightness(img, range_=0.2)`
  - [x] `translate_image(img, range_=0.1)`
  - [x] `augment_image(img, seed=None)`
- [x] Copies original image + generates 4 augmented variants
- [x] Naming format: `{stem}{ext}` for original, `{stem}_aug_{1-4}{ext}` for variants
- [x] Supports `argparse` with `--raw_dir` and `--aug_dir` defaults
- [x] Prints final count per class and total
- [x] Verified dataset counts match 1,200 per class

## 6. Known Issues / Observations
- Source images in NEU dataset have `.jpg` extension (rather than `.bmp`). Script dynamically preserves source extension (`.jpg` / `.bmp` / `.png`) across original and augmented outputs.
- No border artifacts introduced: used `cv2.BORDER_REFLECT_101` for realistic boundary reflections during rotation and translation.
