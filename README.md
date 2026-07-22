# ⚡ EV Fleet Intelligence Platform

> An AI-powered fleet management platform for EV adoption, battery
> analytics, predictive maintenance, and carbon intelligence.

## 📖 Overview

The **EV Fleet Intelligence Platform** is an interactive Streamlit
application that helps fleet operators evaluate fleet electrification,
monitor battery health, predict maintenance requirements, and estimate
environmental impact. The platform combines machine learning with
intuitive dashboards to provide actionable insights for EV fleet
management.

------------------------------------------------------------------------

## ✨ Features

### 🚗 Fleet Electrification Readiness (EFR)

-   Analyze ICE fleets for EV conversion
-   EV suitability scoring
-   Charging infrastructure assessment
-   Fuel cost and savings estimation
-   Vehicle-wise recommendations

### 🔋 Asset Performance Management (APM)

-   State of Health (SOH) prediction
-   Remaining Useful Life (RUL) estimation
-   Thermal risk assessment
-   Battery health classification
-   Live telemetry simulation

### 🛠 Maintenance Optimizer

-   AI-powered maintenance recommendations
-   Battery replacement prediction
-   Cooling system diagnostics
-   Wiring/connector inspection suggestions
-   Preventive maintenance prioritization

### 🌱 Carbon Intelligence Tracker

-   Annual CO₂ savings estimation
-   ICE vs EV emission comparison
-   Carbon credit estimation
-   Sustainability reporting

------------------------------------------------------------------------

## 🧠 Machine Learning

The platform supports prediction models for:

-   State of Health (SOH)
-   Remaining Useful Life (RUL)
-   Thermal Risk
-   Battery Health Classification
-   Cycle Life Prediction
-   Service Recommendation

If trained `.pkl` models are available inside the `models/` folder, they
are loaded automatically. Otherwise, the application falls back to
built-in rule-based logic.

------------------------------------------------------------------------

## 📊 Datasets

This project uses multiple battery datasets including:

-   Battery Health Dataset
-   SOH & RUL Dataset
-   Battery Degradation Dataset
-   Thermal Dataset
-   Cycle Life Dataset
-   Master Battery Training Dataset

Key features include:

-   Voltage
-   Temperature
-   Cycle Count
-   Capacity
-   Capacity Retention
-   Capacity Loss
-   Voltage Drop Rate
-   Temperature Change Rate

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   Streamlit
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib

------------------------------------------------------------------------

## 📂 Project Structure

``` text
EV-Fleet-Intelligence-Platform/
│
├── app.py
├── model_APM_local.ipynb
├── datasets/
├── models/
└── README.md
```

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/yourusername/EV-Fleet-Intelligence-Platform.git
cd EV-Fleet-Intelligence-Platform
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 🎯 Applications

-   EV Fleet Management
-   Logistics & Transportation
-   Smart Cities
-   Battery Analytics
-   Sustainability Monitoring
-   Predictive Maintenance

------------------------------------------------------------------------

## 🔮 Future Improvements

-   IoT Integration
-   Cloud Deployment
-   Real-time Fleet Monitoring
-   Deep Learning Models
-   Route Optimization
-   Mobile Application

------------------------------------------------------------------------

## 👩‍💻 Author

**Bhumi Sisodiya**

B.Tech Computer Science Engineering\
UPES, Dehradun

------------------------------------------------------------------------

## ⭐ Contribute

If you like this project:

-   ⭐ Star the repository
-   🍴 Fork the project
-   💡 Submit improvements

------------------------------------------------------------------------

## 📄 License

This project is intended for educational, research, and demonstration
purposes.

