"""
================================================================================
 LANDSLIDE DETECTION / PREDICTION USING MULTI-SENSOR DATA — END-TO-END PIPELINE
================================================================================
Sensors modeled (all commonly used in real landslide early-warning literature):
    - rainfall_mm            : cumulative/interval rainfall (rain gauge)
    - soil_moisture_pct      : volumetric water content, VWC (capacitive/FBG probe)
    - pore_pressure_kpa      : pore-water pressure (piezometer)
    - slope_angle_deg        : terrain slope at sensor site (static, from DEM)
    - vibration_g            : ground vibration / micro-seismic amplitude (geophone)
    - displacement_mm        : slope surface displacement (extensometer/tiltmeter)
    - temperature_c          : ground/air temperature
    - humidity_pct           : relative humidity

HOW TO USE WITH YOUR OWN DATA
------------------------------
1. Put your CSV at DATA_PATH below. It should have the sensor columns above
   (rename yours to match, or edit FEATURE_COLUMNS) plus a target column
   'landslide' (1 = landslide event, 0 = no event).
2. If DATA_PATH does not exist, the script auto-generates a realistic
   SYNTHETIC dataset (physically-informed, not random noise) so you can run
   the whole pipeline immediately and see it work end to end.
3. Run:  python landslide_detection.py

WHERE TO GET REAL PUBLIC SENSOR DATA (no login-walled scraping needed):
    - NASA Global Landslide Catalog (event records, not raw sensors)
      https://gpm.nasa.gov/landslides/data.html
    - Swiss national soil-moisture + landslide early-warning dataset (Wicki et al.)
      referenced in Landslides journal (Springer) — VWC, soil-water potential,
      ground temperature + meteo data
    - SitkaNet low-cost distributed landslide sensor network (open dataset/paper)
    - Kaggle: search "landslide", "rainfall induced landslide", "slope stability"
    - USGS / Copernicus / Sentinel soil moisture (satellite-based, free API)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "landslide_sensor_data.csv"     # <- point this at your own CSV
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "rainfall_mm", "soil_moisture_pct", "pore_pressure_kpa",
    "slope_angle_deg", "vibration_g", "displacement_mm",
    "temperature_c", "humidity_pct",
]
TARGET_COLUMN = "landslide"


# --------------------------------------------------------------------------
# 1. DATA LOADING (real file if present, else physically-informed synthetic)
# --------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples=4000):
    """
    Builds a synthetic but physically-informed dataset so the pipeline is
    runnable immediately. Landslide risk rises with rainfall, soil moisture,
    pore pressure, slope steepness, vibration and displacement -- mirroring
    the relationships reported in real sensor-network studies (rainfall +
    soil moisture / pore pressure thresholds, geophone/tiltmeter precursors).
    """
    n = n_samples
    rainfall = np.random.gamma(shape=2.0, scale=15, size=n)                 # mm
    soil_moisture = np.clip(np.random.normal(30, 10, n) + rainfall * 0.15, 5, 60)  # %
    pore_pressure = np.clip(np.random.normal(20, 8, n) + soil_moisture * 0.4, 0, 80)  # kPa
    slope_angle = np.clip(np.random.normal(28, 10, n), 5, 60)               # degrees
    vibration = np.clip(np.random.exponential(0.05, n), 0, 2)               # g
    displacement = np.clip(np.random.exponential(2, n) + pore_pressure * 0.05, 0, 50)  # mm
    temperature = np.random.normal(18, 6, n)                                # C
    humidity = np.clip(np.random.normal(65, 15, n) + rainfall * 0.2, 10, 100)  # %

    # Physically-informed risk score -> probability of a landslide event
    risk = (
        0.035 * rainfall +
        0.05 * soil_moisture +
        0.04 * pore_pressure +
        0.06 * slope_angle +
        6.0 * vibration +
        0.12 * displacement -
        6.5
    )
    prob = 1 / (1 + np.exp(-risk / 3))
    landslide = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "soil_moisture_pct": soil_moisture,
        "pore_pressure_kpa": pore_pressure,
        "slope_angle_deg": slope_angle,
        "vibration_g": vibration,
        "displacement_mm": displacement,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "landslide": landslide,
    })
    return df


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading real dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"No file found at {DATA_PATH} -> generating synthetic demo dataset instead.")
        df = generate_synthetic_dataset()
        df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_landslide_data.csv"), index=False)
    return df


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    # Rolling/derived features that matter a lot in real landslide EWS literature
    df["rain_soil_interaction"] = df["rainfall_mm"] * df["soil_moisture_pct"]
    df["pore_pressure_ratio"] = df["pore_pressure_kpa"] / (df["slope_angle_deg"] + 1)
    df["instability_index"] = (
        df["soil_moisture_pct"] * 0.3 +
        df["pore_pressure_kpa"] * 0.3 +
        df["slope_angle_deg"] * 0.2 +
        df["vibration_g"] * 100 * 0.2
    )
    return df


# --------------------------------------------------------------------------
# 3. TRAIN / EVALUATE MULTIPLE MODELS
# --------------------------------------------------------------------------
def train_and_evaluate(df):
    feature_cols = FEATURE_COLUMNS + [
        "rain_soil_interaction", "pore_pressure_ratio", "instability_index"
    ]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "preds": preds,
            "proba": proba,
        }
        print(f"\n{name}")
        print("-" * 40)
        print(classification_report(y_test, preds, zero_division=0))
        print(f"ROC-AUC: {results[name]['roc_auc']:.3f}")

    return results, X_test, y_test, scaler, feature_cols


# --------------------------------------------------------------------------
# 4. VISUALIZATIONS
# --------------------------------------------------------------------------
def plot_results(results, y_test, feature_cols, best_model_name):
    sns.set_style("whitegrid")

    # Model comparison bar chart
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        for name, r in results.items()
    }).T
    ax = metrics_df.plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_title("Model Comparison — Landslide Detection")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # Confusion matrix for best model
    best = results[best_model_name]
    cm = confusion_matrix(y_test, best["preds"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Landslide", "Landslide"],
                yticklabels=["No Landslide", "Landslide"])
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance (if tree-based model available)
    if hasattr(best["model"], "feature_importances_"):
        importances = pd.Series(best["model"].feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=True)
        plt.figure(figsize=(8, 6))
        importances.plot(kind="barh", color="steelblue")
        plt.title(f"Feature Importance — {best_model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
        plt.close()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)

    print(f"\nDataset shape: {df.shape}")
    print(f"Landslide event rate: {df[TARGET_COLUMN].mean():.2%}")

    results, X_test, y_test, scaler, feature_cols = train_and_evaluate(df)

    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_model_name} ({results[best_model_name]['roc_auc']:.3f})")

    plot_results(results, y_test, feature_cols, best_model_name)

    # Save best model + scaler for later inference
    joblib.dump(results[best_model_name]["model"], os.path.join(OUTPUT_DIR, "best_landslide_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    print(f"\nSaved plots + model to ./{OUTPUT_DIR}/")
