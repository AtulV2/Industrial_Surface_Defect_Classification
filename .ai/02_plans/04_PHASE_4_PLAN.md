# Phase 4 Plan — Streamlit UI
**Auto-approved (autopilot mode):** 2026-11-04

## Goal
Build a local Streamlit web app that accepts image uploads, predicts defect class, and shows Grad-CAM overlay.

## UI Layout
- Page: wide layout, title "🔬 Industrial Surface Defect Classifier"
- Sidebar: model info, class list
- Main: file uploader → result header (class + confidence) → two-column display (original | Grad-CAM) → bar chart

## Run command (from project root)
`streamlit run src/app.py`

## Deliverable
`src/app.py`

