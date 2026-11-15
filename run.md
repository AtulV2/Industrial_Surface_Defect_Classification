Simple — just one command. Open a terminal in the project folder and run:

```bash
cd "D:\SysFolders\atulv\Desktop\Project\Industrial_Surface_Defect_Classification"
streamlit run src/app.py
```

Or directly from inside `src`:
```bash
cd "D:\SysFolders\atulv\Desktop\Project\Industrial_Surface_Defect_Classification\src"
streamlit run app.py
```

Streamlit will auto-open your browser at `http://localhost:8501`. Done!

---

### Full Step-by-Step (first time)

**Step 1 — Open terminal in the project folder**
- Right-click the `Industrial_Surface_Defect_Classification` folder → **Open in Terminal**

**Step 2 — (Optional) Install dependencies if not done yet**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch the app**
```bash
streamlit run src/app.py
```

**Step 4 — Use the app**
1. Browser opens automatically at `http://localhost:8501`
2. Click **Browse files** and upload any `.bmp`, `.jpg`, or `.png` surface image
3. See the **predicted defect class + confidence** at the top
4. View **original vs Grad-CAM heatmap** side by side
5. Check the **confidence bar chart** for all 6 classes

**Step 5 — Stop the app**
Press `Ctrl + C` in the terminal.

---

> **Note:** The app automatically resolves the model path whether you run from the project root (`streamlit run src/app.py`) or from inside `src` (`streamlit run app.py`).

---

### If you want to re-train from scratch later:
```bash
# Step 1 — Regenerate augmented data (only needed if raw data changes)
python src/augment.py

# Step 2 — Train
python src/train.py

# Step 3 — Run app
streamlit run src/app.py
```