# Contributing Guide

Thank you for your interest in contributing to this project.

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

## Phase-Gate Development Protocol

This project uses a strict **Phase-Gate execution model** (see `.ai/01_rules/`). All changes must:
1. Be drafted and approved before implementation
2. Be confined strictly to the current phase scope
3. Be audited and logged after execution
4. Be committed to Git after each working sub-step

## Code Standards
- Clear, meaningful variable names
- Inline comments explaining the *why* behind complex logic
- No hardcoded secrets — use `.env` files
- All new dependencies must be added to `requirements.txt`

## Commit Style
```
feat(phase-N): short description of change
fix(phase-N): what was broken and how it was fixed
docs: documentation updates
```

