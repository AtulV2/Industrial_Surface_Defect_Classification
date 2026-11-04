# Phase 1 Plan: Offline Data Augmentation Pipeline

**Phase:** 1 — Augmentation  
**Script:** `src/augment.py`  
**Status:** In Progress (Autopilot Execution)  
**Date:** 2026-11-04  

---

## 1. Objectives & Scope
- Read all images across 6 defect classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) from `data/raw/train/images/{class}/`.
- For each image, copy the original file into `data/augmented/train/{class}/` and generate 4 distinct augmented variants using OpenCV and NumPy.
- Total raw train images: 1,440 (240 per class).
- Total target augmented train images: 7,200 (1,200 per class; 1 original + 4 augmented copies).
- Ensure 100% reproducibility via fixed seeds (`np.random.seed(42)`).

---

## 2. Augmentation Transformations
All transforms implemented via OpenCV (`cv2`) and NumPy (no TensorFlow dependency in this script):
1. **Rotation (`rotate_image`)**:
   - Random angle within $[-\text{angle\_range}, \text{angle\_range}]$ where default is $\pm 15^\circ$.
   - Uses `cv2.getRotationMatrix2D` centered at image midpoint with `cv2.BORDER_REFLECT_101`.
2. **Horizontal Flip (`flip_image`)**:
   - Random horizontal flip with probability $p=0.5$ using `cv2.flip(img, 1)`.
3. **Brightness Adjustment (`adjust_brightness`)**:
   - Random scaling factor $\alpha \in [1.0 - \text{range\_}, 1.0 + \text{range\_}]$ with default $\pm 20\%$.
   - Clipped to $[0, 255]$ with `uint8` casting.
4. **Translation (`translate_image`)**:
   - Random offset $\Delta x \in [-\text{range\_} \cdot W, \text{range\_} \cdot W]$ and $\Delta y \in [-\text{range\_} \cdot H, \text{range\_} \cdot H]$ with default $\pm 10\%$.
   - Applied via `cv2.warpAffine` using `cv2.BORDER_REFLECT_101`.
5. **Composite Augmentation (`augment_image`)**:
   - Executes rotation $\to$ flip $\to$ brightness $\to$ translation in sequence.

---

## 3. CLI & Directory Structure
- Script path: `src/augment.py`
- Arguments via `argparse`:
  - `--raw_dir`: Path to raw training images (default: `data/raw/train/images`).
  - `--aug_dir`: Path to augmented training directory (default: `data/augmented/train`).
  - `--num_aug`: Number of augmented variants per image (default: 4).
  - `--seed`: Random seed for reproducibility (default: 42).
- Class subfolders dynamically created if not present.
- Output reporting: Logs per-class processing progress and final file counts.

---

## 4. Verification Criteria
- Execute `python src/augment.py`.
- Verify each of the 6 classes contains exactly 1,200 images in `data/augmented/train/{class}/`.
- Verify total augmented dataset size is 7,200 images.
- Verify file naming conventions (`{stem}{ext}` for original, `{stem}_aug_{i}{ext}` for variants).

---

## 5. Artifacts & Deliverables
- `src/augment.py`
- `.ai/04_audits/AUDIT_002.md`
- `.ai/03_checkpoints/CHECKPOINT_002.md`
- `.ai/05_chat_logs/LOG_002_phase_1_decisions.md`
- Updated `.ai/CURRENT_STATE.md`
- Git commit: `feat(phase-1): implement offline dataset augmentation with cv2 and numpy`

