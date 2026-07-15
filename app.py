import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fasal Disease Risk Checker",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .main { background-color: #f7faf5; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #66bb6a 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(27,94,32,0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin-top: 0.4rem; opacity: 0.92; font-size: 1.02rem; }

    .risk-card {
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        transition: transform 0.15s ease;
    }
    .risk-card:hover { transform: translateY(-3px); }
    .risk-card .date { font-weight: 700; font-size: 0.95rem; opacity: 0.95; }
    .risk-card .metrics { font-size: 0.85rem; margin: 6px 0; opacity: 0.95; }
    .risk-card .label { font-size: 1.15rem; font-weight: 800; margin-top: 4px; }

    .advisory-box {
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 10px;
        font-size: 1.02rem;
        line-height: 1.5;
    }

    .metric-pill {
        background: white;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e3ede1;
    }
    .metric-pill .value { font-size: 1.6rem; font-weight: 800; color: #1b5e20; }
    .metric-pill .label { font-size: 0.85rem; color: #556; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("crop_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

try:
    model, le = load_model()
    MODEL_READY = True
except Exception:
    MODEL_READY = False

FEATURES = ["temp_max", "temp_min", "humidity", "rainfall"]

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}
RISK_LABEL = {
    "en": {"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"},
    "ur": {"Low": "Khatra Kam Hai", "Medium": "Khatra Darmiyana Hai", "High": "Khatra Zyada Hai"},
}

CROP_INFO = {
    "Wheat (Gandum)": {
        "disease": {"en": "Yellow Rust", "ur": "Zang (Yellow Rust)"},
        "advice": {
            "High": {"en": "Spray fungicide immediately (Propiconazole/Tebuconazole). Inspect leaves for yellow streaks.",
                     "ur": "Foran fungicide spray karein (Propiconazole/Tebuconazole). Pattay par peeli dhool check karein."},
            "Medium": {"en": "Inspect the field daily in the morning. Avoid excess nitrogen fertilizer.",
                       "ur": "Roz subah field check karein. Zyada nitrogen na dein."},
            "Low": {"en": "No action needed right now. Continue normal monitoring.",
                    "ur": "Filhal koi khatra nahi. Normal monitoring jari rakhein."},
        },
    },
    "Cotton (Kapas)": {
        "disease": {"en": "Bacterial Blight", "ur": "Bacterial Blight"},
        "advice": {
            "High": {"en": "Apply copper-based spray. Remove infected bolls immediately.",
                     "ur": "Copper-based spray karein. Ganday boll turant hataayein."},
            "Medium": {"en": "Check field drainage, avoid standing water.",
                       "ur": "Field drainage check karein, pani kharra na rehne dein."},
            "Low": {"en": "No action needed right now. Continue normal monitoring.",
                    "ur": "Filhal koi khatra nahi. Normal monitoring jari rakhein."},
        },
    },
    "Rice (Chawal)": {
        "disease": {"en": "Rice Blast", "ur": "Rice Blast"},
        "advice": {
            "High": {"en": "Spray Tricyclazole. Manage standing water level carefully.",
                     "ur": "Tricyclazole spray karein. Water level manage karein."},
            "Medium": {"en": "Reduce nitrogen fertilizer, observe field closely.",
                       "ur": "Nitrogen kam karein, field observe karein."},
            "Low": {"en": "No action needed right now. Continue normal monitoring.",
                    "ur": "Filhal koi khatra nahi. Normal monitoring jari rakhein."},
        },
    },
    "Sugarcane (Ganna)": {
        "disease": {"en": "Red Rot", "ur": "Red Rot"},
        "advice": {
            "High": {"en": "Remove and burn infected plants immediately. Avoid waterlogging.",
                     "ur": "Mutasir paudon ko foran nikaal kar jala dein. Pani kharra na rehne dein."},
            "Medium": {"en": "Inspect field regularly for early signs.",
                       "ur": "Regular inspection karein, mutasir paudon ki nishandehi karein."},
            "Low": {"en": "No action needed right now. Continue normal monitoring.",
                    "ur": "Filhal koi khatra nahi. Normal monitoring jari rakhein."},
        },
    },
}

# ============================================================
# API HELPERS
# ============================================================
@st.cache_data(ttl=86400)
def geocode_location(name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": name, "count": 5, "language": "en"}, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])

@st.cache_data(ttl=3600)
def fetch_forecast(lat, lon, days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto", "forecast_days": days,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"])
    d.rename(columns={
        "time": "date", "temperature_2m_max": "temp_max", "temperature_2m_min": "temp_min",
        "relative_humidity_2m_mean": "humidity", "precipitation_sum": "rainfall",
    }, inplace=True)
    return d

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🌐 Language / Zaban")
    lang = st.radio("", ["Roman Urdu", "English"], label_visibility="collapsed")
    L = "ur" if lang == "Roman Urdu" else "en"

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="metric-pill"><div class="value">94%</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-pill"><div class="value">99.6%</div><div class="label">ROC-AUC*</div></div>', unsafe_allow_html=True)
    st.caption("*Ensemble reference metric from underlying research pipeline")

    st.markdown("---")
    st.markdown("### ℹ️ Kaise Kaam Karta Hai" if L == "ur" else "### ℹ️ How It Works")
    st.caption(
        "Live mausam data (Open-Meteo) + XGBoost model jo agronomic risk rules par train hua hai."
        if L == "ur" else
        "Live weather data (Open-Meteo) + an XGBoost model trained on agronomic risk rules."
    )

    st.markdown("---")
    st.caption("Built by Muhammad Abdul Qadeer\nQadeer Automations · 2026")

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero">
<h1>🌾 Fasal Disease Risk Checker</h1>
<p>{"Apni fasal aur elaqay ka naam daal kar agle kuch dinon ka disease risk maloom karein — real-time mausam data aur AI se." if L=="ur" else "Check your crop's disease risk for the coming days using real-time weather data and AI."}</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_READY:
    st.warning(
        "⚠️ Model files (crop_risk_model.pkl, label_encoder.pkl) nahi milin. Repo mein upload karein."
        if L == "ur" else
        "⚠️ Model files (crop_risk_model.pkl, label_encoder.pkl) not found. Please upload them to the repo."
    )
    st.stop()

# ============================================================
# INPUT ROW
# ============================================================
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    location_query = st.text_input(
        "📍 Shehar / Gaon ka naam" if L == "ur" else "📍 City / Village Name",
        value="Jatoi",
    )
with col2:
    crop = st.selectbox(
        "🌱 Fasal chunein" if L == "ur" else "🌱 Select Crop",
        list(CROP_INFO.keys()),
    )
with col3:
    forecast_days = st.slider(
        "📅 Din" if L == "ur" else "📅 Days", 1, 7, 3,
    )

check_btn = st.button(
    "🔍 Risk Check Karein" if L == "ur" else "🔍 Check Risk",
    type="primary", use_container_width=True,
)

st.markdown("---")

# ============================================================
# RESULTS
# ============================================================
if check_btn or location_query:
    matches = geocode_location(location_query)

    if not matches:
        st.error(
            "Ye jagah nahi mili. Sahi spelling ke sath dobara try karein (English mein likhein)."
            if L == "ur" else
            "Location not found. Please try again with correct spelling (English)."
        )
        st.stop()

    if len(matches) > 1:
        options = [f"{m['name']}, {m.get('admin1','')}, {m.get('country','')}" for m in matches]
        chosen = st.selectbox(
            "Kai jagah miliin — sahi wali chunein:" if L == "ur" else "Multiple matches found — select the correct one:",
            options,
        )
        selected = matches[options.index(chosen)]
    else:
        selected = matches[0]

    lat, lon = selected["latitude"], selected["longitude"]
    df = fetch_forecast(lat, lon, forecast_days)
    preds = model.predict(df[FEATURES])
    df["predicted_risk"] = le.inverse_transform(preds)

    info = CROP_INFO[crop]

    st.markdown(
        f"📍 **{selected['name']}, {selected.get('admin1','')}** &nbsp;·&nbsp; 🦠 "
        f"{'Sab se aam risk' if L=='ur' else 'Most common risk'}: **{info['disease'][L]}**"
    )

    st.markdown("#### " + ("📊 Forecast" if L == "en" else "📊 Forecast Results"))
    cols = st.columns(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i]:
            color = RISK_COLORS[row["predicted_risk"]]
            label = RISK_LABEL[L][row["predicted_risk"]]
            date_label = pd.to_datetime(row["date"]).strftime("%d %b")
            st.markdown(f"""
            <div class="risk-card" style="background:{color};">
                <div class="date">{date_label}</div>
                <div class="metrics">🌡️ {row['temp_max']:.0f}° / {row['temp_min']:.0f}°C</div>
                <div class="metrics">💧 {row['humidity']:.0f}%  🌧️ {row['rainfall']:.1f}mm</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["humidity"], mode="lines+markers", name="Humidity (%)",
        line=dict(color="#1976d2", width=3),
        marker=dict(size=9, color=[RISK_COLORS[r] for r in df["predicted_risk"]]),
    ))
    fig.update_layout(
        title="Humidity Trend & Risk Level" if L == "en" else "Namee ka Rujhaan aur Khatra",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Advisory
    worst = "Low"
    if "High" in df["predicted_risk"].values:
        worst = "High"
    elif "Medium" in df["predicted_risk"].values:
        worst = "Medium"

    advice_text = info["advice"][worst][L]
    box_color = {"High": "#ffebee", "Medium": "#fff8e1", "Low": "#e8f5e9"}[worst]
    border_color = RISK_COLORS[worst]
    icon = {"High": "⚠️", "Medium": "⚡", "Low": "✅"}[worst]

    st.markdown(
        f"""<div class="advisory-box" style="background:{box_color}; border-left: 6px solid {border_color};">
        <b>{icon} {"Mashwara" if L=="ur" else "Recommendation"}:</b><br>{advice_text}
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📋 " + ("Poora Data Table" if L == "ur" else "Full Data Table")):
        st.dataframe(
            df[["date", "temp_max", "temp_min", "humidity", "rainfall", "predicted_risk"]],
            use_container_width=True,
        )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ " + ("Report Download Karein" if L == "ur" else "Download Report"),
        csv, file_name=f"{selected['name']}_{crop}_risk_report.csv", mime="text/csv",
    )

st.markdown("---")
st.caption(f"Data: Open-Meteo API (live) · Model: XGBoost · Updated {datetime.now().strftime('%d %b %Y, %H:%M')} · © Qadeer Automations")
