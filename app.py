"""
================================================================================
 LANDSLIDE EARLY-WARNING DASHBOARD (Streamlit)
================================================================================
Run with:
    pip install -r requirements.txt
    streamlit run app.py

Expects 'outputs/best_landslide_model.pkl' and 'outputs/scaler.pkl' produced
by landslide_detection.py. If missing, trains a quick model on synthetic
data automatically so the dashboard still runs standalone.

Three modes (pick from the sidebar):
  1. Manual sensor input -> sliders simulate one live reading, big dramatic
     risk reveal + a "what's driving this" explanation.
  2. Upload CSV (batch)   -> score a whole sensor export at once, see all
     alerts + charts in one place, download results.
  3. Live simulation      -> auto-streaming readings so you can watch risk
     rise and fall in real time.
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Landslide Early-Warning Dashboard",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "outputs/best_landslide_model.pkl"
SCALER_PATH = "outputs/scaler.pkl"

FEATURE_COLUMNS = [
    "rainfall_mm", "soil_moisture_pct", "pore_pressure_kpa",
    "slope_angle_deg", "vibration_g", "displacement_mm",
    "temperature_c", "humidity_pct",
]
FEATURE_LABELS = {
    "rainfall_mm": "Rainfall", "soil_moisture_pct": "Soil Moisture",
    "pore_pressure_kpa": "Pore Pressure", "slope_angle_deg": "Slope Angle",
    "vibration_g": "Vibration", "displacement_mm": "Displacement",
    "temperature_c": "Temperature", "humidity_pct": "Humidity",
    "rain_soil_interaction": "Rainfall × Soil Moisture",
    "pore_pressure_ratio": "Pore Pressure Ratio",
    "instability_index": "Instability Index",
}

# Soft, easy-on-the-eyes palette — informative without feeling alarming
RISK_LEVELS = [
    (0.0, 0.25, "LOW", "#3fb572", "🟢"),
    (0.25, 0.5, "MODERATE", "#e0a929", "🟡"),
    (0.5, 0.75, "HIGH", "#e07b39", "🔴"),
    (0.75, 1.01, "CRITICAL", "#d9584f", "🚨"),
]

# --------------------------------------------------------------------------
# GLOBAL STYLE — light theme, bigger type, gentle colors
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; font-size: 17px; }

.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
}
#MainMenu, footer, header {visibility: hidden;}

/* Hero header */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #eaf4ff 0%, #f2fbf5 100%);
    border: 1px solid #dce8f0;
    box-shadow: 0 8px 24px rgba(30,60,90,0.08);
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 900;
    color: #1c2b3a;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #4a6072;
    margin-top: 0.5rem;
    font-size: 1.15rem;
}
.hero-badges span {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #dce8f0;
    color: #35506a;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 12px;
}

/* Section card */
.card {
    background: #ffffff;
    border: 1px solid #e3ebf1;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 16px rgba(30,60,90,0.06);
    margin-bottom: 1.2rem;
}
.card h3, .card h4 { color: #1c2b3a; margin-top: 0; font-size: 1.4rem; }

/* Big dramatic result reveal */
.result-hero {
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 20px;
    border: 2px solid var(--rcolor, #3fb572);
    background: linear-gradient(180deg, var(--rcolor-bg, #eafaf1) 0%, #ffffff 100%);
}
.result-hero .emoji { font-size: 4rem; line-height: 1; }
.result-hero .pct { font-size: 4.2rem; font-weight: 900; color: var(--rcolor, #3fb572); line-height: 1.1; margin: 0.2rem 0; }
.result-hero .level { font-size: 1.6rem; font-weight: 800; color: #1c2b3a; letter-spacing: 0.03em; }
.result-hero .msg { font-size: 1.15rem; color: #4a6072; margin-top: 0.6rem; }

/* KPI metric cards */
.kpi {
    background: #ffffff;
    border: 1px solid #e3ebf1;
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    text-align: left;
    box-shadow: 0 3px 10px rgba(30,60,90,0.05);
}
.kpi .label {
    color: #6b7f92;
    font-size: 0.95rem;
    font-weight: 600;
}
.kpi .value {
    color: #1c2b3a;
    font-size: 2.1rem;
    font-weight: 900;
    margin-top: 3px;
}

/* Alert banner (soft, not alarming) */
.alert-banner {
    border-radius: 16px;
    padding: 1.2rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-weight: 700;
    font-size: 1.2rem;
    border: 2px solid;
    margin-bottom: 1rem;
}
.alert-critical { background: #fdeeed; border-color: #f0b3ae; color: #a83a32; }
.alert-high { background: #fdf1e7; border-color: #f0c79a; color: #a5622a; }
.alert-moderate { background: #fdf7e6; border-color: #f0dd9e; color: #93711d; }
.alert-low { background: #eafaf1; border-color: #a9e3bf; color: #237a4a; }
.alert-banner .emoji { font-size: 2rem; }

.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e3ebf1;
}
section[data-testid="stSidebar"] * { color: #2b3a48 !important; font-size: 1.02rem; }
section[data-testid="stSidebar"] h3 { font-size: 1.35rem !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #3fa9f5, #3fb572);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.55rem 1.1rem;
}
.stButton>button:hover, .stDownloadButton>button:hover { opacity: 0.9; }

/* Bigger labels & headings across the app */
label, .stMarkdown p, .stCaption, p { font-size: 1.05rem !important; color: #33475a; }
h1 { font-size: 2.6rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.6rem !important; }
h4 { font-size: 1.35rem !important; }
.stSlider label, .stRadio label { font-size: 1.1rem !important; font-weight: 600 !important; }
.stSlider [data-baseweb="slider"] { padding-top: 0.4rem; }

/* DataFrame */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #e3ebf1; font-size: 1.05rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#33475a", family="Inter, sans-serif", size=15),
)


def engineer_features(df):
    df = df.copy()
    df["rain_soil_interaction"] = df["rainfall_mm"] * df["soil_moisture_pct"]
    df["pore_pressure_ratio"] = df["pore_pressure_kpa"] / (df["slope_angle_deg"] + 1)
    df["instability_index"] = (
        df["soil_moisture_pct"] * 0.3 +
        df["pore_pressure_kpa"] * 0.3 +
        df["slope_angle_deg"] * 0.2 +
        df["vibration_g"] * 100 * 0.2
    )
    return df


def classify_risk(prob):
    for low, high, label, color, emoji in RISK_LEVELS:
        if low <= prob < high:
            return label, color, emoji
    return "CRITICAL", "#d9584f", "🚨"


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    # Fallback: train an ensemble on synthetic data so the dashboard still
    # works standalone if landslide_detection.py hasn't been run yet.
    n = 3000
    rng = np.random.default_rng(42)
    rainfall = rng.gamma(2.0, 15, n)
    soil_moisture = np.clip(rng.normal(30, 10, n) + rainfall * 0.15, 5, 60)
    pore_pressure = np.clip(rng.normal(20, 8, n) + soil_moisture * 0.4, 0, 80)
    slope_angle = np.clip(rng.normal(28, 10, n), 5, 60)
    vibration = np.clip(rng.exponential(0.05, n), 0, 2)
    displacement = np.clip(rng.exponential(2, n) + pore_pressure * 0.05, 0, 50)
    temperature = rng.normal(18, 6, n)
    humidity = np.clip(rng.normal(65, 15, n) + rainfall * 0.2, 10, 100)
    risk = (0.035*rainfall + 0.05*soil_moisture + 0.04*pore_pressure +
            0.06*slope_angle + 6.0*vibration + 0.12*displacement - 6.5)
    prob = 1 / (1 + np.exp(-risk / 3))
    landslide = rng.binomial(1, prob)

    df = pd.DataFrame({
        "rainfall_mm": rainfall, "soil_moisture_pct": soil_moisture,
        "pore_pressure_kpa": pore_pressure, "slope_angle_deg": slope_angle,
        "vibration_g": vibration, "displacement_mm": displacement,
        "temperature_c": temperature, "humidity_pct": humidity,
        "landslide": landslide,
    })
    df = engineer_features(df)
    feature_cols = FEATURE_COLUMNS + ["rain_soil_interaction", "pore_pressure_ratio", "instability_index"]
    scaler = StandardScaler().fit(df[feature_cols])
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(df[feature_cols]), df["landslide"])
    return model, scaler


def predict(model, scaler, df):
    df = engineer_features(df)
    feature_cols = FEATURE_COLUMNS + ["rain_soil_interaction", "pore_pressure_ratio", "instability_index"]
    X = scaler.transform(df[feature_cols])
    proba = model.predict_proba(X)[:, 1]
    df["risk_probability"] = proba
    levels = df["risk_probability"].apply(classify_risk)
    df["risk_level"] = levels.apply(lambda x: x[0])
    df["risk_color"] = levels.apply(lambda x: x[1])
    df["risk_emoji"] = levels.apply(lambda x: x[2])
    return df


def risk_gauge(prob, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 54, "color": "#1c2b3a"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9fb2c3", "tickfont": {"size": 14, "color": "#6b7f92"}},
            "bar": {"color": color, "thickness": 0.34},
            "bgcolor": "#f2f6fa",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#eafaf1"},
                {"range": [25, 50], "color": "#fdf7e6"},
                {"range": [50, 75], "color": "#fdf1e7"},
                {"range": [75, 100], "color": "#fdeeed"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=25, r=25, t=25, b=15), **PLOTLY_TEMPLATE)
    return fig


def dramatic_result(prob, label, color, emoji):
    bg = {"LOW": "#eafaf1", "MODERATE": "#fdf7e6", "HIGH": "#fdf1e7", "CRITICAL": "#fdeeed"}[label]
    messages = {
        "LOW": "Conditions look stable. No action needed right now.",
        "MODERATE": "Conditions are trending upward. Worth keeping an eye on.",
        "HIGH": "Meaningful instability signs — increase monitoring and get ready to respond.",
        "CRITICAL": "Strong signs of instability — notify your response team.",
    }
    st.markdown(
        f"""
        <div class="result-hero" style="--rcolor:{color}; --rcolor-bg:{bg};">
            <div class="emoji">{emoji}</div>
            <div class="pct">{prob:.0%}</div>
            <div class="level">{label} RISK</div>
            <div class="msg">{messages[label]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if label == "LOW":
        st.balloons()


def alert_banner(level_tuple, prob, prefix=""):
    label, color, emoji = level_tuple if level_tuple else classify_risk(prob)
    css_class = {"CRITICAL": "alert-critical", "HIGH": "alert-high",
                 "MODERATE": "alert-moderate", "LOW": "alert-low"}[label]
    st.markdown(
        f"""<div class="alert-banner {css_class}"><span class="emoji">{emoji}</span>
        <span>{prefix}<b>{label} RISK</b> — {prob:.0%} probability</span></div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label, value):
    st.markdown(f"""<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>""",
                unsafe_allow_html=True)


def risk_pill_html(label, color):
    return f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}66;">{label}</span>'


def feature_importance_chart(model, feature_cols, top_n=8):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.rename(lambda c: FEATURE_LABELS.get(c, c))
    imp = imp.sort_values(ascending=True).tail(top_n)
    fig = px.bar(
        imp, x=imp.values, y=imp.index, orientation="h",
        color=imp.values, color_continuous_scale=["#bcdff0", "#3fa9f5", "#245b8f"],
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE, showlegend=False, coloraxis_showscale=False,
        xaxis_title="Influence on prediction", yaxis_title="", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>⛰️ Landslide Early-Warning Dashboard</h1>
        <p>ML-powered risk monitoring using rainfall, soil moisture, pore pressure, vibration and displacement sensors.</p>
        <div class="hero-badges">
            <span>🌧️ Rain Gauge</span><span>💧 Soil Moisture</span><span>📈 Piezometer</span>
            <span>📡 Geophone</span><span>📏 Extensometer</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, scaler = load_model()
feature_cols_full = FEATURE_COLUMNS + ["rain_soil_interaction", "pore_pressure_ratio", "instability_index"]

with st.sidebar:
    st.markdown("### 🛰️ Control Panel")
    mode = st.radio(
        "Choose a view",
        ["Manual sensor input", "Upload CSV (batch)", "Live simulation"],
    )
    st.markdown("---")
    st.markdown("#### Risk levels")
    for low, high, label, color, emoji in RISK_LEVELS:
        st.markdown(f"{emoji} {risk_pill_html(label, color)} &nbsp; {int(low*100)}–{int(min(high,1)*100)}%",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Model:** {type(model).__name__} ensemble")
    st.caption("Trained on 8 sensor inputs + 3 engineered features. Swap `outputs/*.pkl` to use a model trained on real data.")

# --------------------------------------------------------------------------
# MODE 1: MANUAL INPUT
# --------------------------------------------------------------------------
if mode == "Manual sensor input":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ Simulate a live sensor reading")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 150.0, 20.0)
        soil_moisture = st.slider("💧 Soil moisture (%)", 0.0, 60.0, 30.0)
    with c2:
        pore_pressure = st.slider("📈 Pore pressure (kPa)", 0.0, 80.0, 20.0)
        slope_angle = st.slider("⛰️ Slope angle (°)", 0.0, 60.0, 28.0)
    with c3:
        vibration = st.slider("📡 Vibration (g)", 0.0, 2.0, 0.05)
        displacement = st.slider("📏 Displacement (mm)", 0.0, 50.0, 2.0)
    with c4:
        temperature = st.slider("🌡️ Temperature (°C)", -10.0, 45.0, 18.0)
        humidity = st.slider("💦 Humidity (%)", 0.0, 100.0, 65.0)
    st.markdown('</div>', unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        "rainfall_mm": rainfall, "soil_moisture_pct": soil_moisture,
        "pore_pressure_kpa": pore_pressure, "slope_angle_deg": slope_angle,
        "vibration_g": vibration, "displacement_mm": displacement,
        "temperature_c": temperature, "humidity_pct": humidity,
    }])
    result = predict(model, scaler, input_df).iloc[0]

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        dramatic_result(result["risk_probability"], result["risk_level"], result["risk_color"], result["risk_emoji"])
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Risk meter")
        st.plotly_chart(
            risk_gauge(result["risk_probability"], result["risk_level"], result["risk_color"]),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 What's driving this prediction")
    fi_fig = feature_importance_chart(model, feature_cols_full)
    if fi_fig is not None:
        st.plotly_chart(fi_fig, use_container_width=True)
    else:
        st.caption("Feature importance isn't available for this model type.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# MODE 2: CSV UPLOAD (BATCH)
# --------------------------------------------------------------------------
elif mode == "Upload CSV (batch)":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Batch-score sensor readings from a CSV")
    st.caption(f"Required columns: {', '.join(FEATURE_COLUMNS)}  (optionally: site_id, timestamp)")
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            scored = predict(model, scaler, raw)
            n_alerts = (scored["risk_level"].isin(["HIGH", "CRITICAL"])).sum()

            m1, m2, m3, m4 = st.columns(4)
            with m1: kpi_card("Records scored", f"{len(scored):,}")
            with m2: kpi_card("Active alerts", int(n_alerts))
            with m3: kpi_card("Avg. risk", f"{scored['risk_probability'].mean():.0%}")
            with m4: kpi_card("Max risk", f"{scored['risk_probability'].max():.0%}")

            st.write("")
            if n_alerts > 0:
                alert_banner(None, scored["risk_probability"].max(), prefix=f"{n_alerts} site(s) flagged — ")
            else:
                alert_banner(("LOW", "#3fb572", "🟢"), scored["risk_probability"].max(), prefix="All clear — ")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Scored readings")
            display_cols = (["site_id"] if "site_id" in scored.columns else []) + \
                            (["timestamp"] if "timestamp" in scored.columns else []) + \
                            FEATURE_COLUMNS + ["risk_probability", "risk_level"]
            styled = scored[display_cols].sort_values("risk_probability", ascending=False)
            st.dataframe(
                styled.style.apply(
                    lambda row: [f"background-color: {classify_risk(row['risk_probability'])[1]}22"] * len(row),
                    axis=1,
                ).format({"risk_probability": "{:.0%}"}),
                use_container_width=True,
            )
            st.download_button(
                "⬇ Download scored results",
                scored.to_csv(index=False).encode(),
                "landslide_risk_scored.csv",
                "text/csv",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            colX, colY = st.columns(2)
            with colX:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                fig = px.histogram(
                    scored, x="risk_probability", color="risk_level",
                    color_discrete_map={l: c for _, _, l, c, _ in RISK_LEVELS},
                    nbins=20, title="Risk probability distribution",
                )
                fig.update_layout(**PLOTLY_TEMPLATE, legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with colY:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 🧠 What's driving risk overall")
                fi_fig = feature_importance_chart(model, feature_cols_full)
                if fi_fig is not None:
                    st.plotly_chart(fi_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload a CSV to see batch predictions and alerts.")

# --------------------------------------------------------------------------
# MODE 3: LIVE SIMULATION
# --------------------------------------------------------------------------
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📡 Live sensor stream simulation")
    st.caption("Generates a new reading every few seconds to demonstrate real-time alerting. "
               "Replace `simulate_reading()` with a real API/MQTT/serial feed for production use.")
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        speed = st.slider("Update interval (s)", 1, 10, 2)
    with cc2:
        run = st.checkbox("▶ Start live stream")
    st.markdown('</div>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=FEATURE_COLUMNS + ["risk_probability", "risk_level", "t"])

    placeholder = st.empty()

    def simulate_reading(t):
        phase = (np.sin(t / 10) + 1) / 2
        return {
            "rainfall_mm": 10 + phase * 100 + np.random.normal(0, 5),
            "soil_moisture_pct": 20 + phase * 35 + np.random.normal(0, 3),
            "pore_pressure_kpa": 10 + phase * 60 + np.random.normal(0, 4),
            "slope_angle_deg": 30 + np.random.normal(0, 2),
            "vibration_g": max(0, phase * 1.2 + np.random.normal(0, 0.1)),
            "displacement_mm": phase * 30 + np.random.normal(0, 2),
            "temperature_c": 18 + np.random.normal(0, 2),
            "humidity_pct": min(100, 50 + phase * 40 + np.random.normal(0, 5)),
        }

    if run:
        for t in range(60):
            reading = simulate_reading(t)
            df_r = pd.DataFrame([reading])
            scored = predict(model, scaler, df_r).iloc[0]
            new_row = {**reading, "risk_probability": scored["risk_probability"],
                       "risk_level": scored["risk_level"], "t": t}
            st.session_state.history = pd.concat(
                [st.session_state.history, pd.DataFrame([new_row])], ignore_index=True
            ).tail(50)

            with placeholder.container():
                latest = st.session_state.history.iloc[-1]
                alert_banner(classify_risk(latest["risk_probability"]), latest["risk_probability"], prefix=f"t={t}s — ")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                fig = px.line(st.session_state.history, x="t", y="risk_probability",
                               title="Rolling risk probability", markers=True)
                fig.update_traces(line_color="#3fa9f5", marker=dict(size=7))
                fig.add_hline(y=0.5, line_dash="dash", line_color="#e07b39", annotation_text="High")
                fig.add_hline(y=0.75, line_dash="dash", line_color="#d9584f", annotation_text="Critical")
                fig.update_layout(**PLOTLY_TEMPLATE, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    st.session_state.history[FEATURE_COLUMNS + ["risk_probability", "risk_level"]].tail(10),
                    use_container_width=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(speed)
    else:
        st.info("Check the box above to start the simulated live stream.")
        if not st.session_state.history.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(st.session_state.history.tail(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)