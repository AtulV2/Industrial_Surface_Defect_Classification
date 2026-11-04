# LOG_002: Phase 1 Key Decisions

**Date:** 2026-11-04  
**Phase:** 1 — Augmentation  
**Topic:** Offline Augmentation Design and Execution  

---

## Decisions & Rationale

1. **Standalone Script with Pure OpenCV / NumPy**:
   - Requirement strictly specified avoiding TensorFlow for the augmentation script.
   - Using pure OpenCV (`cv2`) and NumPy ensures lightweight execution, fast warpAffine/flip/brightness transformations, and no framework overhead.

2. **Boundary Padding Method**:
   - During random rotation ($\pm 15^\circ$) and translation ($\pm 10\%$), image pixels shift leaving unfilled boundaries.
   - We selected `cv2.BORDER_REFLECT_101` over zero-padding (`BORDER_CONSTANT`). Black border bands could otherwise become artificial edge features that bias defect classification. Reflection provides seamless metallurgical surface continuity.

3. **Handling Dataset File Extensions**:
   - While documentation referenced `.bmp`, actual files in `data/raw/` are `.jpg`.
   - The script was engineered using `Path.stem` and `Path.suffix` to preserve native extensions seamlessly across both original and augmented files without unnecessary trans-coding.

4. **Preservation of Original Images**:
   - Originals were copied directly via `shutil.copy2` to preserve metadata and avoid any lossy re-encoding artifacts.

