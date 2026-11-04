# Audit & Verification Protocol

## Every Audit Must Include
1. **Files Added** — list every new file created with its path
2. **Files Modified** — list every file changed and what changed
3. **Dependencies Added** — any new packages added to `requirements.txt`
4. **Execution Result** — stdout/stderr from running the code or tests
5. **Phase Requirements Check** — tick off every requirement from the approved plan
6. **Known Issues / Risks** — any deviations or concerns for the next phase

## Verification Steps (per phase)
- Phase 1 (Data): Confirm dataset loads, shapes correct, augmentation applied
- Phase 2 (Model): Confirm model summary, training completes, `.keras` saved
- Phase 3 (Grad-CAM): Confirm heatmap generated, overlay returned correctly
- Phase 4 (UI): Confirm Streamlit app launches, upload → predict → Grad-CAM visible

