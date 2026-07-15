# 🌾 Crop Disease Risk Checker

A real-time crop disease risk prediction system built for Pakistani farmers — check the disease risk for **Wheat, Cotton, Rice, or Sugarcane** in any city or village, using live weather data and machine learning.

🔗 **Live App:** _[Add your Streamlit Cloud link here after deployment]_

---

## 📌 Overview

Most crop disease outbreaks (rust, blight, blast, red rot) are strongly linked to weather conditions — specifically humidity, temperature, and rainfall patterns. This project combines:

- **Live weather forecasting** (Open-Meteo API — free, no key required)
- **Agronomic risk rules** derived from known plant pathology patterns
- **An XGBoost classifier** trained on 2 years of historical weather data
- **A bilingual (Roman Urdu / English) Streamlit interface** so it's usable by any farmer, not just technical users

Anyone can type in their city/village name, select their crop, and get an easy-to-understand risk forecast with actionable advice — in Roman Urdu.

---

## ✨ Features

- 🔍 **Search any location** — not limited to a fixed list of cities (uses Open-Meteo Geocoding API)
- 🌱 **4 major crops supported** — Wheat, Cotton, Rice, Sugarcane, each with disease-specific advisory
- 📅 **1–7 day forecast** with live, real-time weather data
- 🗣️ **Roman Urdu / English toggle** for accessibility
- 📊 **Visual risk cards + trend chart** for quick understanding
- 💡 **Actionable recommendations** (not just a risk label — tells the farmer what to do)
- ⬇️ **Downloadable CSV report**

---

## 🧠 How the Model Works

1. **Data collection:** 2 years of historical daily weather data (max/min temperature, humidity, rainfall) pulled from the Open-Meteo Archive API.
2. **Labeling:** Since no public labeled crop-disease dataset exists for this region, risk labels (Low/Medium/High) were generated using established agronomic rules — e.g., fungal disease risk rises sharply with humidity above 70–75% combined with moderate temperatures (15–32°C), and further increases with rainfall.
3. **Model:** An XGBoost classifier was trained on these features, achieving **~94% accuracy** on a held-out test set (stratified split).
4. **Explainability:** SHAP (SHapley Additive exPlanations) was used to verify that humidity and temperature are the dominant drivers of predicted risk — consistent with real plant pathology.
5. **Deployment:** At prediction time, the app fetches **live** weather forecasts (not historical/static data) for whatever location the user enters, so the risk shown is always current.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| ML Model | XGBoost |
| Explainability | SHAP |
| Weather Data | Open-Meteo API (Archive + Forecast + Geocoding) |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |
| Development | Google Colab |

---

## 📂 Repository Structure

```
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── crop_risk_model.pkl     # Trained XGBoost model
├── label_encoder.pkl       # Label encoder for risk classes
└── README.md                # Project documentation
```

---

## 🚀 Running Locally

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Overall Accuracy | 94% |
| High Risk Precision | 0.83 |
| Medium Risk Precision | 0.85 |
| Low Risk Precision | 1.00 |

---

## 🔮 Future Improvements

- Incorporate satellite/soil moisture data for higher precision
- Add SMS/WhatsApp alert delivery for farmers without internet access
- Expand crop and disease coverage
- Multi-year data to better capture seasonal disease cycles

---

## 👤 Author

**Muhammad Abdul Qadeer**
AI & Automation Specialist — Qadeer Automations
