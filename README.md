# Landslide Detection ML Pipeline + Alert Dashboard

## What's in this folder
```
landslide_project/
├── landslide_detection.py   # Trains models, evaluates, saves best model + plots
├── app.py                   # Streamlit alert dashboard (manual / CSV / live modes)
├── requirements.txt         # Python dependencies
└── outputs/                 # Pre-generated demo results (from synthetic data)
    ├── best_landslide_model.pkl
    ├── scaler.pkl
    ├── synthetic_landslide_data.csv
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Setup in VS Code

1. **Unzip** this folder and open it in VS Code (`File > Open Folder`).

2. **Create a virtual environment** (recommended). Open a terminal in VS Code
   (`` Ctrl+` ``) and run:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If VS Code prompts you to select a Python interpreter, choose the one
   inside `venv`.

4. **Run the ML pipeline** (optional — a model is already included in
   `outputs/`, but rerun this if you add your own dataset or want fresh
   plots):
   ```bash
   python landslide_detection.py
   ```

5. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```
   This opens automatically in your browser at `http://localhost:8501`.
   If it doesn't, click the link printed in the terminal.

## Using your own dataset instead of synthetic data

1. Get a public sensor/rainfall/landslide dataset (Kaggle, NASA Global
   Landslide Catalog, a national soil-moisture network, etc.).
2. Rename its columns to match (or edit `FEATURE_COLUMNS` in
   `landslide_detection.py` to match your columns instead):
   ```
   rainfall_mm, soil_moisture_pct, pore_pressure_kpa, slope_angle_deg,
   vibration_g, displacement_mm, temperature_c, humidity_pct, landslide
   ```
3. Save it as `landslide_sensor_data.csv` in this same folder.
4. Rerun `python landslide_detection.py` — it detects the file and trains
   on your real data instead of synthetic, overwriting the model in
   `outputs/`.
5. Rerun `streamlit run app.py` (or just refresh the browser tab) to pick
   up the new model.

## Dashboard modes (in the sidebar)
- **Manual sensor input** — sliders simulate one live reading, instant risk
  gauge + color-coded alert.
- **Upload CSV (batch)** — score a whole sensor export at once; alert
  table, risk histogram, downloadable results.
- **Live simulation** — auto-streams readings so you can watch risk climb
  from LOW to CRITICAL and see the alert fire. Swap `simulate_reading()`
  in `app.py` for a real API/MQTT feed to go live in production.

## Troubleshooting
- **`ModuleNotFoundError`** → make sure your venv is activated and you ran
  `pip install -r requirements.txt` inside it.
- **Column mismatch error on CSV upload** → your CSV needs all 8 columns
  listed in `FEATURE_COLUMNS` above (case-sensitive names).
- **Streamlit doesn't open a browser** → manually visit the URL printed in
  the terminal (usually `http://localhost:8501`).

