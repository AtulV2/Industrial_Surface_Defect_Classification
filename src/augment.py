"""
augment.py — Offline Data Augmentation for NEU Surface Defect Dataset
======================================================================
Reads original grayscale images from data/raw/train/images/{class}/,
copies each original to data/augmented/train/{class}/, then generates
4 augmented variants per image using seeded random transforms.

Output layout (Option B — originals + augmented in one folder):
    data/augmented/train/
        crazing/
            crazing_1.bmp          <- original copy
            crazing_1_aug_1.bmp    <- augmented variant 1
            ...
            crazing_1_aug_4.bmp    <- augmented variant 4
        inclusion/ ...

Usage:
    python src/augment.py
    python src/augment.py --raw_dir data/raw/train/images --aug_dir data/augmented/train --copies 4
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Augmentation helpers ───────────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle_range: float = 15.0) -> np.ndarray:
    """Rotate image by a random angle within ±angle_range degrees."""
    h, w = img.shape[:2]
    angle = np.random.uniform(-angle_range, angle_range)
    # Rotation matrix around the image center
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=angle, scale=1.0)
    # BORDER_REFLECT_101 fills empty pixels with mirrored content (avoids black borders)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def flip_image(img: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally with given probability."""
    if np.random.random() < prob:
        return cv2.flip(img, 1)   # 1 = horizontal flip
    return img


def adjust_brightness(img: np.ndarray, range_: float = 0.2) -> np.ndarray:
    """
    Multiply pixel values by a random factor in [1-range_, 1+range_].
    Works on grayscale or BGR images. Clips to valid [0, 255] range.
    """
    factor = np.random.uniform(1.0 - range_, 1.0 + range_)
    # Convert to float32 for multiplication, then clip and cast back
    brightened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return brightened


def translate_image(img: np.ndarray, range_: float = 0.1) -> np.ndarray:
    """
    Translate image by a random offset up to range_ * image dimension.
    Uses reflection border to avoid black edges.
    """
    h, w = img.shape[:2]
    tx = np.random.uniform(-range_ * w, range_ * w)
    ty = np.random.uniform(-range_ * h, range_ * h)
    # Affine translation matrix
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply all four augmentation transforms in sequence.
    Each call produces a unique variant (driven by global np.random state).
    """
    img = rotate_image(img)
    img = flip_image(img)
    img = adjust_brightness(img)
    img = translate_image(img)
    return img


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_augmentation(raw_dir: str, aug_dir: str, copies: int) -> None:
    """
    For each class subfolder in raw_dir:
      1. Copy the original image into aug_dir/{class}/
      2. Generate `copies` augmented variants into the same folder.
    """
    raw_path = Path(raw_dir)
    aug_path = Path(aug_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_path}")

    class_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in: {raw_path}")

    print(f"\n{'='*60}")
    print(f"  NEU-DET Offline Augmentation (seed={SEED})")
    print(f"  Source : {raw_path}")
    print(f"  Output : {aug_path}")
    print(f"  Copies : {copies} augmented variants per original")
    print(f"{'='*60}\n")

    total_written = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class_dir = aug_path / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(class_dir.glob("*.bmp")) +
                             list(class_dir.glob("*.jpg")) +
                             list(class_dir.glob("*.png")))

        class_count = 0

        for img_path in image_files:
            # Read image — cv2 reads BMP natively; grayscale mode
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Could not read: {img_path.name} — skipping")
                continue

            stem = img_path.stem          # e.g. "crazing_1"
            suffix = img_path.suffix      # e.g. ".bmp"

            # 1. Save original copy
            orig_dest = out_class_dir / img_path.name
            cv2.imwrite(str(orig_dest), img)
            class_count += 1

            # 2. Generate augmented copies
            for i in range(1, copies + 1):
                aug_img = augment_image(img)
                aug_name = f"{stem}_aug_{i}{suffix}"
                cv2.imwrite(str(out_class_dir / aug_name), aug_img)
                class_count += 1

        total_written += class_count
        print(f"  [{class_name:>20}]  originals: {len(image_files):>4}  "
              f"total written: {class_count:>5}")

    print(f"\n{'='*60}")
    print(f"  Augmentation complete — {total_written} images written")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Resolve project root relative to this script's location
    script_dir = Path(__file__).resolve().parent          # src/
    project_root = script_dir.parent                      # project root

    parser = argparse.ArgumentParser(
        description="Offline augmentation for NEU Surface Defect dataset"
    )
    parser.add_argument(
        "--raw_dir",
        default=str(project_root / "data" / "raw" / "train" / "images"),
        help="Path to raw training images directory (contains class subfolders)"
    )
    parser.add_argument(
        "--aug_dir",
        default=str(project_root / "data" / "augmented" / "train"),
        help="Output path for augmented images (class subfolders will be created)"
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=4,
        help="Number of augmented copies to generate per original image (default: 4)"
    )
    args = parser.parse_args()

    run_augmentation(args.raw_dir, args.aug_dir, args.copies)
