"""
EV Fleet Intelligence Platform — Demo App
==========================================
Wraps the scoring/classification logic from the project's notebooks
(model_APM.ipynb, Battery_Service_Recommendation_MODEL.ipynb) into an
interactive Streamlit dashboard for a demo/recording.

Modules (narrative order): EFR -> APM -> Maintenance Optimiser -> Carbon Tracker

NOTES ON MODEL LOADING
-----------------------
If you export your trained models from the notebooks (joblib.dump) and
drop the .pkl files into a `models/` folder next to this file, the app
will automatically use the REAL trained models instead of the formula
based fallback below. Expected filenames (from model_APM.ipynb):
    models/soh_model.pkl
    models/rul_model.pkl
    models/thermal_model.pkl
    models/cycle_life_model.pkl
    models/health_model.pkl
    models/battery_service_model.pkl
    models/service_label_encoder.pkl

Until those exist, the app uses the same formulas already written in
your notebooks (thermal risk bands, health class bands, the 35/25/15/15/10
battery health score, and the service recommendation rules) so it runs
standalone with zero external data files — good enough for a live demo.

RUN LOCALLY
-----------
    pip install streamlit pandas numpy joblib
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EV Fleet Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"


# ----------------------------------------------------------------------
# Optional real-model loading (falls back to formulas if not present)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    if os.path.isdir(MODELS_DIR):
        for name in [
            "soh_model", "rul_model", "thermal_model", "cycle_life_model",
            "health_model", "battery_service_model", "service_label_encoder",
        ]:
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(path):
                try:
                    models[name] = joblib.load(path)
                except Exception:
                    pass
    return models


MODELS = load_models()
USING_REAL_MODELS = len(MODELS) > 0


# ----------------------------------------------------------------------
# Formula-based fallbacks (mirrors the notebook logic)
# ----------------------------------------------------------------------
def get_thermal_risk(temp):
    if temp >= 45:
        return "High"
    elif temp >= 35:
        return "Medium"
    return "Low"


THERMAL_SCORE_MAP = {"Low": 100, "Medium": 60, "High": 20}


def get_health_class(soh_percent):
    if soh_percent >= 90:
        return "Healthy"
    elif soh_percent >= 75:
        return "Moderate"
    elif soh_percent >= 60:
        return "Degraded"
    return "Critical"


HEALTH_SCORE_MAP = {"Healthy": 100, "Moderate": 75, "Degraded": 50, "Critical": 20}


def estimate_soh_rul(cycle, voltage, temperature, capacity, capacity_loss,
                      capacity_retention, voltage_drop_rate,
                      temperature_change_rate, cycle_percentage):
    """Formula-based stand-in for soh_model / rul_model when no .pkl present."""
    if "soh_model" in MODELS:
        X = pd.DataFrame([{
            "cycle": cycle, "voltage": voltage, "temperature": temperature,
            "capacity": capacity, "capacity_loss": capacity_loss,
            "capacity_retention": capacity_retention,
            "voltage_drop_rate": voltage_drop_rate,
            "temperature_change_rate": temperature_change_rate,
            "cycle_percentage": cycle_percentage,
        }])
        soh = float(MODELS["soh_model"].predict(X)[0])
        rul = float(MODELS["rul_model"].predict(X)[0]) if "rul_model" in MODELS else max(0, 1000 - cycle)
        return soh, rul

    # Fallback heuristic: degrade with cycle count and capacity loss
    soh = max(0.0, min(1.0, capacity_retention / 100 - capacity_loss / 200))
    rul = max(0.0, 1000 * (1 - cycle_percentage / 100) - capacity_loss * 5)
    return soh, rul


def battery_health_score(soh_percent, rul, thermal_risk, health_class, capacity_retention, max_rul=1000):
    rul_score = float(np.clip(rul / max_rul * 100, 0, 100))
    thermal_score = THERMAL_SCORE_MAP[thermal_risk]
    health_score = HEALTH_SCORE_MAP[health_class]
    score = (
        0.35 * soh_percent
        + 0.25 * rul_score
        + 0.15 * thermal_score
        + 0.15 * health_score
        + 0.10 * capacity_retention
    )
    return score, rul_score


def diagnose_service(soh_percent, rul, thermal_risk, capacity_retention, voltage_drop_rate, score):
    """Root-cause-aware recommendation — distinguishes battery-end-of-life from
    wiring/connector faults and cooling/thermal issues, instead of defaulting
    every low score to 'Battery Replacement'."""
    if rul < 50 or capacity_retention < 55:
        return ("Battery Replacement",
                "Capacity retention / remaining useful life too low to recover — battery has reached end-of-life.")
    if voltage_drop_rate > 0.6 and capacity_retention >= 70:
        return ("Wiring / Connector Inspection",
                "Voltage drop is high but capacity retention is still healthy — points to a connection fault, not the battery.")
    if thermal_risk == "High" and capacity_retention >= 70:
        return ("Cooling System Inspection",
                "Temperature risk is high but capacity retention is still healthy — points to a thermal management fault, not the battery.")
    if thermal_risk == "High" and capacity_retention < 70:
        return ("Immediate Service (Battery + Cooling)",
                "Both thermal risk and capacity retention are poor — needs urgent combined inspection.")
    if 55 <= capacity_retention < 75 or 50 <= rul < 150:
        return ("Battery Reconditioning / Cell Balancing",
                "Moderate degradation detected — reconditioning or cell balancing may restore performance without full replacement.")
    if score >= 90:
        return ("No Service", "All indicators within healthy range.")
    if score >= 75:
        return ("Routine Inspection", "Minor wear detected — schedule a routine check.")
    return ("Preventive Maintenance", "General preventive maintenance recommended.")


RISK_COLOR = {
    "No Service": "🟢", "Routine Inspection": "🟢", "Preventive Maintenance": "🟡",
    "Battery Reconditioning / Cell Balancing": "🟡", "Wiring / Connector Inspection": "🟠",
    "Cooling System Inspection": "🟠", "Immediate Service (Battery + Cooling)": "🟠",
    "Battery Replacement": "🔴",
}


# ----------------------------------------------------------------------
# Synthetic fleet (for the fleet-overview table — ~100 vehicles)
# ----------------------------------------------------------------------
@st.cache_data
def generate_fleet(n=100, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        cycle_percentage = rng.uniform(5, 95)
        capacity_retention = 100 - cycle_percentage * rng.uniform(0.5, 0.9)
        capacity_loss = 100 - capacity_retention
        temperature = rng.uniform(20, 50)
        voltage_drop_rate = rng.uniform(0, 1)
        soh, rul = estimate_soh_rul(
            cycle=cycle_percentage * 10, voltage=3.7, temperature=temperature,
            capacity=2.0, capacity_loss=capacity_loss,
            capacity_retention=capacity_retention, voltage_drop_rate=voltage_drop_rate,
            temperature_change_rate=rng.uniform(0, 1), cycle_percentage=cycle_percentage,
        )
        soh_percent = soh * 100 if soh <= 1 else soh
        thermal_risk = get_thermal_risk(temperature)
        health_class = get_health_class(soh_percent)
        score, _ = battery_health_score(soh_percent, rul, thermal_risk, health_class, capacity_retention)
        action, reason = diagnose_service(soh_percent, rul, thermal_risk, capacity_retention, voltage_drop_rate, score)
        rows.append({
            "Vehicle ID": f"EV-{i:03d}", "SOH %": round(soh_percent, 1),
            "RUL (cycles)": round(rul), "Temp °C": round(temperature, 1),
            "Thermal Risk": thermal_risk, "Health Class": health_class,
            "Health Score": round(score, 1), "Recommended Action": action,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Synthetic ICE fleet (for EFR — which of our petrol/diesel vehicles
# could switch to EV, decided automatically per vehicle)
# ----------------------------------------------------------------------
@st.cache_data
def generate_ice_fleet(n=100, avg_ev_range=250, charging_points=15, subsidy_eligible=True, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        fuel_type = rng.choice(["Petrol", "Diesel"], p=[0.6, 0.4])
        daily_km = float(rng.uniform(20, 400))
        route_type = rng.choice(["Fixed short route", "Fixed long route", "Variable/long-haul"],
                                 p=[0.5, 0.3, 0.2])
        kmpl = rng.uniform(10, 20) if fuel_type == "Petrol" else rng.uniform(14, 24)
        fuel_price = 105 if fuel_type == "Petrol" else 92  # INR/l, rough
        monthly_fuel_cost = (daily_km * 30 / kmpl) * fuel_price
        ev_energy_cost = (daily_km * 30 * 0.18) * 8  # 0.18 kWh/km, INR 8/kWh
        monthly_savings = monthly_fuel_cost - ev_energy_cost

        # Suitability scoring
        route_fit = float(np.clip(100 - (daily_km / avg_ev_range) * 100, 0, 100))
        savings_fit = float(np.clip((monthly_savings / max(monthly_fuel_cost, 1)) * 100, 0, 100))

        # Route-level charging accessibility: overall fleet infra scaled down by how
        # exposed the route is — a short fixed route near a depot can lean on one
        # charger; a variable/long-haul route needs charging spread along the way,
        # which the fleet's charging points don't guarantee.
        overall_infra = float(np.clip((charging_points / max(n * 0.1, 1)) * 100, 0, 100))
        route_accessibility_multiplier = {
            "Fixed short route": 1.0, "Fixed long route": 0.6, "Variable/long-haul": 0.3,
        }[route_type]
        infra_fit = overall_infra * route_accessibility_multiplier

        route_variability_penalty = {"Fixed short route": 0, "Fixed long route": 10,
                                      "Variable/long-haul": 30}[route_type]
        subsidy_bonus = 8 if subsidy_eligible else 0

        suitability = float(np.clip(
            0.45 * route_fit + 0.30 * savings_fit + 0.15 * infra_fit
            - route_variability_penalty + subsidy_bonus, 0, 100
        ))

        if suitability >= 65:
            recommendation = "Switch to EV"
        elif suitability >= 40:
            recommendation = "Reconsider Later"
        else:
            recommendation = "Keep ICE"

        rows.append({
            "Vehicle ID": f"ICE-{i:03d}", "Fuel Type": fuel_type,
            "Daily Distance (km)": round(daily_km), "Route Type": route_type,
            "Charging Access %": round(infra_fit),
            "Est. Monthly Fuel Cost (₹)": round(monthly_fuel_cost),
            "Est. Monthly Savings if EV (₹)": round(monthly_savings),
            "Suitability Score": round(suitability, 1),
            "Recommendation": recommendation,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
badge = "using trained models" if USING_REAL_MODELS else "demo mode — formula-based (drop .pkl files in models/ to use real models)"

SECTIONS = [
    ("efr", "1️⃣", "Fleet Electrification Readiness"),
    ("apm", "2️⃣", "Asset Performance Management"),
    ("maint", "3️⃣", "Maintenance Optimiser"),
    ("carbon", "4️⃣", "Carbon Intelligence Tracker"),
]

if "active_section" not in st.session_state:
    st.session_state.active_section = "efr"

# ---- sidebar styling: turn the nav buttons into a clean vertical menu ----
st.markdown("""
<style>
section[data-testid="stSidebar"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    margin-bottom: 4px;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    border-left: 3px solid #ff4b4b;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚡ EV Fleet\nIntelligence Platform")
    st.caption(badge)
    st.markdown("---")
    for key, icon, label in SECTIONS:
        is_active = st.session_state.active_section == key
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_section = key
            st.rerun()

active = st.session_state.active_section
st.title("⚡ EV Fleet Intelligence Platform")

# ---------------- EFR ----------------
if active == "efr":
    st.header("Fleet Electrification Readiness (EFR)")
    st.write(
        "Given a fleet of 100 petrol/diesel vehicles, which ones should switch to EV? "
        "Every vehicle below is scored automatically — no manual review needed."
    )

    c1, c2 = st.columns(2)
    with c1:
        avg_ev_range = st.slider("Average EV range available for replacement (km)", 50, 600, 250)
        charging_points = st.number_input("Charging points available", 0, 500, 15)
    with c2:
        subsidy_eligible = st.checkbox("Fleet eligible for FAME-II / PM E-DRIVE subsidy", value=True)

    ice_df = generate_ice_fleet(100, avg_ev_range, charging_points, subsidy_eligible)

    n_switch = (ice_df["Recommendation"] == "Switch to EV").sum()
    n_later = (ice_df["Recommendation"] == "Reconsider Later").sum()
    n_keep = (ice_df["Recommendation"] == "Keep ICE").sum()
    total_savings = ice_df.loc[ice_df["Recommendation"] == "Switch to EV", "Est. Monthly Savings if EV (₹)"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Switch to EV now", n_switch)
    m2.metric("Reconsider later", n_later)
    m3.metric("Keep as ICE", n_keep)
    m4.metric("Est. combined monthly savings", f"₹{total_savings:,.0f}")

    def highlight_efr(row):
        color = {"Switch to EV": "#1e3d2f", "Reconsider Later": "#3d3419",
                 "Keep ICE": "#2a2a2a"}.get(row["Recommendation"], "")
        text = {"Switch to EV": "#a9e6bd", "Reconsider Later": "#f0dd93",
                "Keep ICE": "#c9c9c9"}.get(row["Recommendation"], "")
        return [f"background-color: {color}; color: {text}"] * len(row)

    st.subheader("All 100 vehicles — automatic recommendation")
    sort_order = {"Switch to EV": 0, "Reconsider Later": 1, "Keep ICE": 2}
    ice_df_sorted = ice_df.assign(_rank=ice_df["Recommendation"].map(sort_order)) \
                           .sort_values(["_rank", "Suitability Score"], ascending=[True, False]) \
                           .drop(columns="_rank")
    st.dataframe(ice_df_sorted.style.apply(highlight_efr, axis=1), use_container_width=True, height=400)

    st.caption(
        "Score = 45% route fit (daily distance vs. available EV range) + 30% fuel-cost savings "
        "+ 15% route-level charging accessibility (fleet infra scaled down for long/variable routes), "
        "adjusted for route variability and subsidy eligibility. "
        "≥65 → Switch to EV, 40–64 → Reconsider Later, <40 → Keep ICE."
    )

# ---------------- APM ----------------
if active == "apm":
    st.header("Asset Performance Management (APM)")
    st.subheader("Fleet Overview")
    fleet_df = generate_fleet(100)

    def highlight_risk(row):
        color = {
            "No Service": "#1e3d2f", "Routine Inspection": "#1e3d2f",
            "Battery Reconditioning / Cell Balancing": "#3d3419",
            "Preventive Maintenance": "#3d3419",
            "Wiring / Connector Inspection": "#3d2a19",
            "Cooling System Inspection": "#3d2a19",
            "Immediate Service (Battery + Cooling)": "#3d2a19",
            "Battery Replacement": "#3d1f24",
        }.get(row["Recommended Action"], "")
        text = {
            "No Service": "#a9e6bd", "Routine Inspection": "#a9e6bd",
            "Battery Reconditioning / Cell Balancing": "#f0dd93",
            "Preventive Maintenance": "#f0dd93",
            "Wiring / Connector Inspection": "#f0b877",
            "Cooling System Inspection": "#f0b877",
            "Immediate Service (Battery + Cooling)": "#f0b877",
            "Battery Replacement": "#f2a3ae",
        }.get(row["Recommended Action"], "")
        return [f"background-color: {color}; color: {text}"] * len(row)

    st.dataframe(fleet_df.style.apply(highlight_risk, axis=1), use_container_width=True, height=300)

    st.subheader("Drill into a single vehicle")
    vehicle_id = st.selectbox("Select vehicle", fleet_df["Vehicle ID"])
    vrow = fleet_df[fleet_df["Vehicle ID"] == vehicle_id].iloc[0]

    st.write("Adjust telemetry to see live predictions update:")
    c1, c2, c3 = st.columns(3)
    with c1:
        cycle = st.slider("Cycle count", 0, 1000, int(vrow["RUL (cycles)"] * 0 + 300))
        voltage = st.slider("Voltage (V)", 2.5, 4.2, 3.7)
        temperature = st.slider("Temperature (°C)", 15.0, 60.0, float(vrow["Temp °C"]))
    with c2:
        capacity = st.slider("Capacity (Ah)", 0.5, 3.0, 2.0)
        capacity_loss = st.slider("Capacity loss (%)", 0.0, 60.0, float(100 - vrow["SOH %"]))
        capacity_retention = st.slider("Capacity retention (%)", 20.0, 100.0, float(vrow["SOH %"]))
    with c3:
        voltage_drop_rate = st.slider("Voltage drop rate", 0.0, 1.0, 0.2)
        temperature_change_rate = st.slider("Temp change rate", 0.0, 1.0, 0.2)
        cycle_percentage = st.slider("Cycle % of rated life", 0.0, 100.0, 100 - float(vrow["SOH %"]))

    soh, rul = estimate_soh_rul(cycle, voltage, temperature, capacity, capacity_loss,
                                 capacity_retention, voltage_drop_rate, temperature_change_rate,
                                 cycle_percentage)
    soh_percent = soh * 100 if soh <= 1 else soh
    thermal_risk = get_thermal_risk(temperature)
    health_class = get_health_class(soh_percent)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SOH", f"{soh_percent:.1f}%")
    m2.metric("RUL", f"{rul:.0f} cycles")
    m3.metric("Thermal Risk", thermal_risk)
    m4.metric("Health Class", health_class)

    # stash for the Maintenance tab
    st.session_state["apm_result"] = dict(
        vehicle_id=vehicle_id, soh_percent=soh_percent, rul=rul,
        thermal_risk=thermal_risk, health_class=health_class,
        capacity_retention=capacity_retention, voltage_drop_rate=voltage_drop_rate,
    )

# ---------------- Maintenance ----------------
if active == "maint":
    st.header("Maintenance Optimiser")
    if "apm_result" not in st.session_state:
        st.info("Open the APM tab first and select a vehicle.")
    else:
        r = st.session_state["apm_result"]
        score, rul_score = battery_health_score(
            r["soh_percent"], r["rul"], r["thermal_risk"], r["health_class"], r["capacity_retention"]
        )
        action, reason = diagnose_service(
            r["soh_percent"], r["rul"], r["thermal_risk"], r["capacity_retention"],
            r["voltage_drop_rate"], score,
        )

        st.subheader(f"Vehicle: {r['vehicle_id']}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Battery Health Score", f"{score:.1f} / 100")
        with c2:
            st.markdown(f"### {RISK_COLOR.get(action, '')} Recommended Action: **{action}**")
            st.caption(reason)

        st.caption("Score = 35% SOH + 25% RUL + 15% Thermal + 15% Health Class + 10% Capacity Retention")

        st.subheader("Fleet-wide priority queue")
        priority_order = ["Battery Replacement", "Immediate Service (Battery + Cooling)",
                           "Cooling System Inspection", "Wiring / Connector Inspection",
                           "Battery Reconditioning / Cell Balancing", "Preventive Maintenance",
                           "Routine Inspection", "No Service"]
        fleet_df = generate_fleet(100)
        fleet_df["_rank"] = fleet_df["Recommended Action"].apply(lambda a: priority_order.index(a))
        st.dataframe(
            fleet_df.sort_values("_rank").drop(columns="_rank").head(15),
            use_container_width=True,
        )

# ---------------- Carbon Tracker ----------------
if active == "carbon":
    st.header("Net Zero Carbon Intelligence Tracker")
    c1, c2 = st.columns(2)
    with c1:
        n_ev = st.number_input("Number of EVs in fleet", 1, 1000, 100)
        km_per_day = st.slider("Average km/day per vehicle", 10, 500, 120)
        ev_kwh_per_km = st.slider("EV energy use (kWh/km)", 0.05, 0.5, 0.18)
    with c2:
        ice_kmpl = st.slider("Equivalent ICE fuel efficiency (km/l)", 5, 25, 12)
        grid_emission_factor = st.slider("Grid emission factor (kgCO2/kWh)", 0.3, 1.2, 0.82)
        fuel_emission_factor = st.slider("Fuel emission factor (kgCO2/l)", 2.0, 3.0, 2.31)

    annual_km = n_ev * km_per_day * 365
    ev_emissions_kg = annual_km * ev_kwh_per_km * grid_emission_factor
    ice_emissions_kg = (annual_km / ice_kmpl) * fuel_emission_factor
    co2_saved_tonnes = (ice_emissions_kg - ev_emissions_kg) / 1000
    credits_estimate = co2_saved_tonnes  # 1 carbon credit ≈ 1 tonne CO2e (simplified)

    m1, m2, m3 = st.columns(3)
    m1.metric("Annual CO2 saved", f"{co2_saved_tonnes:,.0f} tCO2e")
    m2.metric("vs. equivalent ICE fleet", f"{ice_emissions_kg/1000:,.0f} tCO2e")
    m3.metric("Est. carbon credits", f"{credits_estimate:,.0f}")

    st.caption(
        "Simplified estimate for demo purposes — real carbon-credit issuance follows India's "
        "Carbon Credit Trading Scheme (CCTS) methodology under BEE."
    )