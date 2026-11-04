"""
Industrial Surface Defect Classification - Offline Augmentation Script
Phase 1: Generates augmented variations using OpenCV and NumPy.
"""

import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import cv2

# Set random seed at the top for reproducibility
np.random.seed(42)


def rotate_image(img: np.ndarray, angle_range: float = 15.0) -> np.ndarray:
    """Randomly rotate the image within [-angle_range, +angle_range] degrees."""
    angle = np.random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated


def flip_image(img: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """Randomly flip the image horizontally with probability `prob`."""
    if np.random.rand() < prob:
        return cv2.flip(img, 1)  # 1 indicates horizontal flip
    return img


def adjust_brightness(img: np.ndarray, range_: float = 0.2) -> np.ndarray:
    """Multiply pixel values by a random factor within [1 - range_, 1 + range_]."""
    factor = np.random.uniform(1.0 - range_, 1.0 + range_)
    adjusted = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted


def translate_image(img: np.ndarray, range_: float = 0.1) -> np.ndarray:
    """Randomly translate the image up to range_ * dimensions along x and y."""
    h, w = img.shape[:2]
    dx = np.random.uniform(-range_ * w, range_ * w)
    dy = np.random.uniform(-range_ * h, range_ * h)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return translated


def augment_image(img: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Applies the 4 transformations in sequence:
    1. Rotation (±15°)
    2. Horizontal Flip (50% probability)
    3. Brightness Adjustment (±20%)
    4. Translation (±10% in x and y)
    """
    if seed is not None:
        np.random.seed(seed)

    x = rotate_image(img, angle_range=15.0)
    x = flip_image(x, prob=0.5)
    x = adjust_brightness(x, range_=0.2)
    x = translate_image(x, range_=0.1)
    return x


def process_dataset(raw_dir: str, aug_dir: str, num_aug: int = 4, seed: int = 42) -> dict:
    """
    Reads all raw images per class, copies the original, and generates
    `num_aug` augmented copies. Returns a dict mapping class names to final counts.
    """
    np.random.seed(seed)

    raw_path = Path(raw_dir)
    aug_path = Path(aug_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    aug_path.mkdir(parents=True, exist_ok=True)

    # Defect classes: subfolders in raw_dir
    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: d.name)

    counts = {}

    print("==================================================")
    print(" Industrial Surface Defect Dataset Augmentation")
    print("==================================================")
    print(f"Source Directory: {raw_path.resolve()}")
    print(f"Target Directory: {aug_path.resolve()}")
    print(f"Augmentations per image: {num_aug}")
    print(f"Classes found ({len(class_dirs)}): {[d.name for d in class_dirs]}")
    print("--------------------------------------------------")

    valid_extensions = {".bmp", ".jpg", ".jpeg", ".png"}

    for cdir in class_dirs:
        class_name = cdir.name
        dst_class_dir = aug_path / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in cdir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        image_files.sort(key=lambda f: f.name)

        print(f"Processing class '{class_name}' ({len(image_files)} source images)...")

        total_saved = 0
        for idx, img_file in enumerate(image_files):
            stem = img_file.stem
            ext = img_file.suffix

            # 1. Copy original file
            dst_orig = dst_class_dir / f"{stem}{ext}"
            shutil.copy2(img_file, dst_orig)
            total_saved += 1

            # 2. Read image for augmentation
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {img_file}")

            # 3. Generate augmented variants
            for aug_i in range(1, num_aug + 1):
                aug_img = augment_image(img)
                dst_aug = dst_class_dir / f"{stem}_aug_{aug_i}{ext}"
                cv2.imwrite(str(dst_aug), aug_img)
                total_saved += 1

        counts[class_name] = total_saved
        print(f"  -> Finished '{class_name}': {total_saved} images written to {dst_class_dir.resolve()}")

    print("--------------------------------------------------")
    print("Final Counts per Class:")
    total_all = 0
    for cname, cnt in counts.items():
        print(f"  - {cname:20s}: {cnt:5d} images")
        total_all += cnt
    print(f"Total Images in Augmented Dataset: {total_all}")
    print("==================================================")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate augmented dataset for surface defect classification.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw/train/images",
        help="Path to raw training images directory."
    )
    parser.add_argument(
        "--aug_dir",
        type=str,
        default="data/augmented/train",
        help="Path to augmented training images directory."
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=4,
        help="Number of augmented copies to generate per image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    process_dataset(
        raw_dir=args.raw_dir,
        aug_dir=args.aug_dir,
        num_aug=args.num_aug,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
