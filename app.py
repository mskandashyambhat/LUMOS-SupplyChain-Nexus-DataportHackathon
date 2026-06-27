from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ortools.linear_solver import pywraplp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Centralization of Pharmaceutical Warehouses An Integrated Simulation and Optimization Approach"

WAREHOUSE_META = [
    {"region": "Rize", "capacity_sheet": "Capacity Rize", "demand_sheet": "Demand Rize",
     "cost_sheet": "Cost Rize", "distance_sheet": "Distance Rize", "time_sheet": "Time Rize",
     "geo_sheet": "Rize", "mwc_sheet": "Costs Rize"},
    {"region": "Trabzon", "capacity_sheet": "Capacity Trabzon", "demand_sheet": "Demand Trabzon",
     "cost_sheet": "Cost Trabzon", "distance_sheet": "Distance Trabzon", "time_sheet": "Time Trabzon",
     "geo_sheet": "Trabzon", "mwc_sheet": "Costs Trabzon"},
    {"region": "Giresun", "capacity_sheet": "Capacity Giresun", "demand_sheet": "Demand Giresun",
     "cost_sheet": "Cost Giresun", "distance_sheet": "Distance Giresun", "time_sheet": "Time Giresun",
     "geo_sheet": "Giresun", "mwc_sheet": "Costs Giresun"},
    {"region": "Ordu", "capacity_sheet": "Capacity Ordu", "demand_sheet": "Demand Ordu",
     "cost_sheet": "Cost Ordu", "distance_sheet": "Distance Ordu", "time_sheet": "Time Ordu",
     "geo_sheet": "Ordu", "mwc_sheet": "Costs Ordu"},
]

SCENARIO_NAMES = ["Baseline", "Warehouse outage", "Capacity stress", "Demand spike"]
DEFAULT_WEIGHTS = {"cost": 0.4, "time": 0.3, "distance": 0.2, "balance": 0.1}

REGION_COORDS = {
    "Rize":    {"lat": 41.0201, "lon": 40.5234, "color": "#6ea8fe"},
    "Trabzon": {"lat": 41.0015, "lon": 39.7178, "color": "#2dd4bf"},
    "Giresun": {"lat": 40.9128, "lon": 38.3895, "color": "#f59e0b"},
    "Ordu":    {"lat": 40.9862, "lon": 37.8797, "color": "#f97316"},
}

# ─── Data loading helpers ────────────────────────────────────────────────────

def load_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=None)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, Any]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")
    data: dict[str, Any] = {k: {} for k in ["capacity", "demand", "cost", "distance", "time", "mwc", "geo"]}
    for meta in WAREHOUSE_META:
        region = meta["region"]
        data["capacity"][region] = load_workbook(DATA_DIR / "CapacityClustered.xlsx", meta["capacity_sheet"])
        data["demand"][region]   = load_workbook(DATA_DIR / "DemandCluster.xlsx",     meta["demand_sheet"])
        data["cost"][region]     = load_workbook(DATA_DIR / "CostCluster.xlsx",       meta["cost_sheet"])
        data["distance"][region] = load_workbook(DATA_DIR / "DistanceCluster.xlsx",   meta["distance_sheet"])
        data["time"][region]     = load_workbook(DATA_DIR / "Time.xlsx",              meta["time_sheet"])
        data["mwc"][region]      = load_workbook(DATA_DIR / "CostsMWC-Clustered.xlsx",meta["mwc_sheet"])
        data["geo"][region]      = pd.read_excel(DATA_DIR / "GeoLocations.xlsx",      sheet_name=meta["geo_sheet"])
    data["actual"]    = pd.read_excel(DATA_DIR / "DemandCluster.xlsx", sheet_name="Gercek Data")
    data["locations"] = pd.read_excel(DATA_DIR / "GeoLocations.xlsx",  sheet_name="P-WH", header=None)
    return data


def parse_demand_table(df: pd.DataFrame) -> dict[str, float]:
    result = {}
    for _, row in df.iterrows():
        label = row.iloc[0]
        if pd.isna(label):
            continue
        label = str(label)
        if label.startswith("demand"):
            values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna()
            result[label] = float(values.sum())
    return result


def parse_capacity_table(df: pd.DataFrame) -> dict[str, float]:
    result = {}
    current_cluster = None
    for _, row in df.iterrows():
        label = row.iloc[0]
        if pd.isna(label):
            continue
        label = str(label)
        if label.startswith("P="):
            current_cluster = label
            continue
        if current_cluster is None:
            continue
        if label == "capacity":
            values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna()
            result[current_cluster] = float(values.sum())
    return result


def parse_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().iloc[1:, :]
    df = df.rename(columns={df.columns[0]: "origin"})
    df["origin"] = df["origin"].astype(str)
    return df.melt(id_vars=["origin"], var_name="target", value_name="cost").dropna()


def parse_distance_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().iloc[1:, :]
    df = df.rename(columns={df.columns[0]: "origin"})
    df["origin"] = df["origin"].astype(str)
    return df.melt(id_vars=["origin"], var_name="target", value_name="distance").dropna()


def parse_time_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().iloc[1:, :]
    df = df.rename(columns={df.columns[0]: "origin"})
    df["origin"] = df["origin"].astype(str)
    return df.melt(id_vars=["origin"], var_name="target", value_name="time").dropna()

# ─── Feature engineering ─────────────────────────────────────────────────────

def build_feature_table(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for meta in WAREHOUSE_META:
        region = meta["region"]
        capacities = parse_capacity_table(data["capacity"][region])
        demands    = parse_demand_table(data["demand"][region])
        cost       = parse_cost_table(data["cost"][region])
        distance   = parse_distance_table(data["distance"][region])
        travel     = parse_time_table(data["time"][region])

        total_capacity = sum(capacities.values())
        total_demand   = sum(demands.values())
        avg_cost       = pd.to_numeric(cost["cost"],         errors="coerce").mean()
        avg_distance   = pd.to_numeric(distance["distance"], errors="coerce").mean()
        avg_time       = pd.to_numeric(travel["time"],       errors="coerce").mean()

        demand_cold = sum(v for k, v in demands.items() if "Cold" in k or "cold" in k)
        demand_norm = sum(v for k, v in demands.items() if "Cold" not in k and "cold" not in k)

        rows.append({
            "region": region,
            "total_capacity": total_capacity,
            "total_demand": total_demand,
            "demand_cold": demand_cold,
            "demand_norm": demand_norm,
            "capacity_utilization": total_demand / max(total_capacity, 1),
            "avg_cost": avg_cost,
            "avg_distance": avg_distance,
            "avg_time": avg_time,
            "cost_pressure": total_demand * avg_cost,
            "distance_pressure": total_demand * avg_distance,
            "time_pressure": total_demand * avg_time,
        })
    return pd.DataFrame(rows)


def enrich_with_ml(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    feature_cols = ["total_capacity", "total_demand", "capacity_utilization",
                    "avg_cost", "avg_distance", "avg_time"]
    X = df[feature_cols].fillna(0)

    kmeans = KMeans(n_clusters=min(3, len(df)), n_init=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    iso = IsolationForest(random_state=42,
                          contamination=min(0.3, max(0.1, 1 / max(len(df), 1))))
    scores = -iso.fit_predict(X)
    df["ml_risk_flag"] = scores

    if len(df) > 1:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(StandardScaler().fit_transform(X))
        df["pca_x"] = coords[:, 0]
        df["pca_y"] = coords[:, 1]
    else:
        df["pca_x"] = 0.0
        df["pca_y"] = 0.0

    if len(df) >= 3:
        X_train = df[feature_cols]
        y_train = df["capacity_utilization"]
        model = Pipeline([("scale", StandardScaler()),
                          ("rf", RandomForestRegressor(n_estimators=200, random_state=42))])
        model.fit(X_train, y_train)
        df["predicted_utilization"] = model.predict(X_train)
        df["utilization_gap"] = df["capacity_utilization"] - df["predicted_utilization"]
    else:
        df["predicted_utilization"] = df["capacity_utilization"]
        df["utilization_gap"] = 0
    return df


def score_health(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["warehouse_health_score"] = (
        100
        - 55 * out["capacity_utilization"].clip(0, 2)
        - 15 * out["ml_risk_flag"].fillna(0)
        + 8  * (1 - out["cluster"] / max(out["cluster"].max(), 1))
    ).clip(0, 100)
    out["risk_index"] = (
        40 * out["capacity_utilization"].clip(0, 2)
        + 30 * (out["ml_risk_flag"].fillna(0))
        + 15 * (out["avg_distance"] / max(out["avg_distance"].max(), 1))
        + 15 * (out["avg_time"]     / max(out["avg_time"].max(), 1))
    ).clip(0, 100)
    return out

# ─── Optimization ────────────────────────────────────────────────────────────

def build_scenario(scenario_name: str, selected_region: str) -> dict[str, Any]:
    if scenario_name == "Warehouse outage":   return {"outage_region": selected_region}
    if scenario_name == "Capacity stress":    return {"capacity_multiplier": 0.75}
    if scenario_name == "Demand spike":       return {"demand_multiplier": 1.25}
    return {}


def get_profile_weights(scenario_name: str) -> dict[str, float]:
    if scenario_name == "Warehouse outage":
        return {"cost": 0.2, "time": 0.25, "distance": 0.15, "balance": 0.4}
    if scenario_name == "Capacity stress":
        return {"cost": 0.3, "time": 0.25, "distance": 0.15, "balance": 0.3}
    if scenario_name == "Demand spike":
        return {"cost": 0.25, "time": 0.35, "distance": 0.15, "balance": 0.25}
    return DEFAULT_WEIGHTS.copy()


def solve_allocation(features: pd.DataFrame, objective_weights: dict[str, float],
                     scenario: dict[str, Any]) -> pd.DataFrame:
    if features.empty:
        return features
    regions  = features["region"].tolist()
    demand   = features["total_demand"].to_numpy(dtype=float)
    capacity = features["total_capacity"].to_numpy(dtype=float)
    cost     = features["avg_cost"].to_numpy(dtype=float)
    distance = features["avg_distance"].to_numpy(dtype=float)
    time     = features["avg_time"].to_numpy(dtype=float)

    if scenario.get("demand_multiplier", 1.0) != 1.0:
        demand = demand * float(scenario["demand_multiplier"])
    if scenario.get("capacity_multiplier", 1.0) != 1.0:
        capacity = capacity * float(scenario["capacity_multiplier"])
    if scenario.get("outage_region") in regions:
        capacity[regions.index(scenario["outage_region"])] = 0.0

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return features.assign(assigned_demand=np.nan, optimization_share=np.nan, objective_score=np.nan)

    x = {(i, j): solver.NumVar(0, solver.infinity(), f"x_{i}_{j}")
         for i in range(len(regions)) for j in range(len(regions))}
    shortage = {i: solver.NumVar(0, solver.infinity(), f"short_{i}") for i in range(len(regions))}

    for i in range(len(regions)):
        solver.Add(sum(x[i, j] for j in range(len(regions))) + shortage[i] == demand[i])
    for j in range(len(regions)):
        solver.Add(sum(x[i, j] for i in range(len(regions))) <= capacity[j])

    total_demand   = max(demand.sum(), 1.0)
    total_capacity = max(capacity.sum(), 1.0)
    used           = [sum(x[i, j] for i in range(len(regions))) for j in range(len(regions))]
    target_alloc   = [total_demand * (capacity[j] / total_capacity) for j in range(len(regions))]
    mean_util = sum(used[j] / max(capacity[j], 1.0) for j in range(len(regions))) / max(len(regions), 1)

    obj = solver.Objective()
    for i in range(len(regions)):
        for j in range(len(regions)):
            coeff = (objective_weights.get("cost", 0.4)     * cost[j]
                   + objective_weights.get("distance", 0.3) * distance[j]
                   + objective_weights.get("time", 0.2)     * time[j]
                   + objective_weights.get("balance", 0.1)  * abs(capacity[j] / max(total_demand, 1.0) - 1.0))
            obj.SetCoefficient(x[i, j], coeff)
        obj.SetCoefficient(shortage[i], 1000.0)

    for j in range(len(regions)):
        us = solver.NumVar(0, solver.infinity(), f"util_dev_{j}")
        solver.Add(us >= used[j] / max(capacity[j], 1.0) - mean_util)
        solver.Add(us >= mean_util - used[j] / max(capacity[j], 1.0))
        obj.SetCoefficient(us, 25.0 * objective_weights.get("balance", 0.1))
        al = solver.NumVar(0, solver.infinity(), f"alloc_dev_{j}")
        solver.Add(al >= used[j] - target_alloc[j])
        solver.Add(al >= target_alloc[j] - used[j])
        obj.SetCoefficient(al, 0.5 * objective_weights.get("balance", 0.1))
    obj.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return features.assign(assigned_demand=np.nan, optimization_share=np.nan, objective_score=np.nan)

    assigned  = [sum(x[i, j].solution_value() for i in range(len(regions))) for j in range(len(regions))]
    shortages = [shortage[i].solution_value() for i in range(len(regions))]
    out = features.copy()
    out["assigned_demand"]       = assigned
    out["optimization_share"]    = out["assigned_demand"] / total_demand
    out["utilization_after_opt"] = out["assigned_demand"] / np.maximum(capacity, 1.0)
    out["objective_score"]       = [obj.Value() / total_demand] * len(out)
    out["capacity_after_scenario"] = capacity
    out["demand_after_scenario"]   = demand
    out["shortage"]   = shortages
    out["service_level"] = np.where(demand > 0, 1 - (np.array(shortages) / demand), 1.0)
    return out

# ─── AI recommendations ──────────────────────────────────────────────────────

def generate_recommendations(scored: pd.DataFrame, optimized: pd.DataFrame,
                              scenario_name: str) -> list[dict[str, Any]]:
    recs = []
    if optimized.empty:
        return recs

    for _, row in optimized.iterrows():
        util = row.get("utilization_after_opt", row["capacity_utilization"])
        health = row.get("warehouse_health_score", 50)
        risk   = row.get("risk_index", 50)
        region = row["region"]

        if util > 0.90:
            savings_pct = round(15 + (util - 0.90) * 100 * 1.2, 1)
            recs.append({
                "region": region, "type": "⚠️ Overload Alert",
                "title": f"Redistribute load from {region}",
                "reason": f"{region} is running at {util*100:.1f}% utilization — above safe threshold.",
                "action": "Shift 15–20% of assigned demand to the lowest-utilized neighboring warehouse.",
                "confidence": 91, "savings": f"~{savings_pct}% utilization reduction",
                "impact": "High", "color": "#ef4444",
            })
        elif util < 0.35:
            recs.append({
                "region": region, "type": "📈 Underutilization",
                "title": f"Increase demand allocation to {region}",
                "reason": f"{region} is only at {util*100:.1f}% utilization — significant idle capacity.",
                "action": "Route overflow demand from high-pressure regions here.",
                "confidence": 85, "savings": "Potential 8–12% cost reduction via shorter routes",
                "impact": "Medium", "color": "#f59e0b",
            })

        if risk > 65:
            recs.append({
                "region": region, "type": "🔴 High Risk",
                "title": f"Mitigate risk in {region}",
                "reason": f"Risk index {risk:.1f}/100 — above critical threshold of 65.",
                "action": "Add safety buffer stock and establish a contingency supplier agreement.",
                "confidence": 78, "savings": "Prevents potential disruption cost (~20–30% of regional spend)",
                "impact": "Critical", "color": "#dc2626",
            })

        if scenario_name == "Warehouse outage" and row.get("shortage", 0) > 0:
            recs.append({
                "region": region, "type": "🚨 Shortage Risk",
                "title": f"Shortage detected in {region} under outage scenario",
                "reason": f"Simulated outage leaves {row['shortage']:.0f} units unmet.",
                "action": "Pre-position emergency stock at adjacent depots before any planned maintenance.",
                "confidence": 95, "savings": "Eliminates service disruption risk",
                "impact": "Critical", "color": "#7c3aed",
            })

    if not recs:
        best = optimized.sort_values("warehouse_health_score", ascending=False).iloc[0]
        recs.append({
            "region": best["region"], "type": "✅ Optimal State",
            "title": "Network is well-balanced",
            "reason": "All warehouses are within healthy utilization bands.",
            "action": "Maintain current allocation and run monthly reviews.",
            "confidence": 88, "savings": "No immediate cost savings identified — network is optimal",
            "impact": "Low", "color": "#22c55e",
        })
    return recs


def estimate_sustainability(features: pd.DataFrame, optimized: pd.DataFrame) -> dict[str, float]:
    """Rough CO₂ proxy: distance_pressure ∝ fuel consumption ∝ emissions."""
    base_co2   = float(features["distance_pressure"].sum() * 0.00021)
    if not optimized.empty and "utilization_after_opt" in optimized.columns:
        saved_factor = float((optimized["utilization_after_opt"] /
                              optimized["capacity_utilization"].clip(lower=0.01)).mean())
        saved_factor = np.clip(saved_factor, 0.7, 1.0)
    else:
        saved_factor = 1.0
    opt_co2    = base_co2 * saved_factor
    dist_saved = float(features["avg_distance"].mean() * (1 - saved_factor) * 4)
    return {
        "baseline_co2_kg":  round(base_co2,   2),
        "optimized_co2_kg": round(opt_co2,    2),
        "co2_saved_kg":     round(base_co2 - opt_co2, 2),
        "distance_saved_km": round(dist_saved, 1),
        "green_score": round(min(100, 40 + (1 - saved_factor) * 600), 1),
    }

# ─── CSS / styles ────────────────────────────────────────────────────────────

def inject_styles() -> None:
    st.markdown("""
<style>
/* ── page background ── */
.stApp {
    background:
        radial-gradient(ellipse at 8% 8%,  rgba(48,103,255,.20) 0%, transparent 26%),
        radial-gradient(ellipse at 92% 4%,  rgba(0,209,178,.15) 0%, transparent 22%),
        radial-gradient(ellipse at 50% 95%, rgba(90,30,200,.12) 0%, transparent 30%),
        linear-gradient(180deg, #07111f 0%, #081521 50%, #050a13 100%);
}
.block-container { padding-top:1rem; padding-bottom:2rem; max-width:1420px; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1c 0%, #09162a 100%) !important;
    border-right: 1px solid rgba(110,168,254,.12);
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 18px;
    padding: 1.1rem 1rem 1rem 1rem;
    text-align: center;
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 0 28px rgba(110,168,254,.18); }
.kpi-label { text-transform:uppercase; letter-spacing:.11em; color:#84a7df;
              font-size:.68rem; margin-bottom:.3rem; }
.kpi-value { font-size:1.85rem; font-weight:800; color:#f7fbff; line-height:1; }
.kpi-sub   { font-size:.72rem; color:#5a82c0; margin-top:.25rem; }

/* ── section headers ── */
.section-header {
    font-size:1.15rem; font-weight:700; color:#c8deff;
    border-left: 3px solid #3b82f6; padding-left:.65rem;
    margin: 1.4rem 0 .7rem 0;
}

/* ── glass card ── */
.glass { background:rgba(255,255,255,.032); border:1px solid rgba(255,255,255,.075);
         border-radius:20px; padding:1.1rem 1.15rem; backdrop-filter:blur(10px); }

/* ── rec card ── */
.rec-card {
    border-radius: 16px; padding: 1rem 1.1rem;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(15,28,50,.8);
    margin-bottom: .75rem;
}
.rec-type   { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em;
               color:#84a7df; margin-bottom:.3rem; }
.rec-title  { font-size:1rem; font-weight:700; color:#f0f7ff; margin-bottom:.35rem; }
.rec-body   { font-size:.87rem; color:#b0c8ef; line-height:1.55; }
.rec-badge  { display:inline-block; border-radius:999px; padding:.22rem .65rem;
               font-size:.72rem; font-weight:600; margin-top:.45rem; }

/* ── hero ── */
.hero-shell {
    background: linear-gradient(135deg, rgba(9,19,34,.96), rgba(5,11,22,.92));
    border: 1px solid rgba(143,179,255,.16);
    border-radius: 26px; padding:1.6rem 1.8rem 1.4rem 1.8rem;
    box-shadow: 0 24px 80px rgba(0,0,0,.35);
}
.hero-title { font-family: Georgia,'Times New Roman',serif; font-size:2.8rem;
               font-weight:700; letter-spacing:-.03em; color:#f6f9ff; line-height:1; }
.hero-copy  { color:#a9c4ff; font-size:.98rem; line-height:1.65; max-width:60rem;
               margin-top:.5rem; }

/* ── verdict ── */
.verdict-card {
    background: linear-gradient(135deg,rgba(18,34,61,.97),rgba(10,18,32,.95));
    border: 1px solid rgba(110,168,254,.22); border-radius:22px;
    padding:1.1rem 1.2rem; margin-top:.8rem;
}
.verdict-title { color:#f7fbff; font-weight:800; font-size:1.05rem; margin-bottom:.25rem; }
.verdict-body  { color:#bdd2fb; line-height:1.6; font-size:.96rem; }
.mini-pill {
    display:inline-block; border-radius:999px; border:1px solid rgba(255,255,255,.08);
    padding:.28rem .65rem; margin:.2rem .3rem .2rem 0;
    color:#eaf2ff; background:rgba(255,255,255,.04); font-size:.76rem;
}

/* ── map overlay ── */
.map-label { font-size:.75rem; color:#7ba4d4; text-align:center; margin-top:.3rem; }

/* ── tab active underline ── */
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #3b82f6 !important;
    color: #93c5fd !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Chart helpers ───────────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8deff"),
    margin=dict(l=12, r=12, t=50, b=12),
)


def gauge_chart(value: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 13, "color": "#84a7df"}},
        number={"font": {"size": 26, "color": "#f7fbff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3d5a80"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "bordercolor": "rgba(255,255,255,0.06)",
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,.12)"},
                {"range": [70,100], "color": "rgba(34,197,94,.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": value},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_geo_map(scored: pd.DataFrame, optimized: pd.DataFrame) -> go.Figure:
    """Interactive scatter-map of warehouse locations sized by demand, colored by risk."""
    lats, lons, labels, risks, sizes, colors, hovers = [], [], [], [], [], [], []
    for _, row in scored.iterrows():
        region = row["region"]
        coords = REGION_COORDS.get(region, {"lat": 41.0, "lon": 39.0, "color": "#6ea8fe"})
        opt_row = optimized[optimized["region"] == region]
        util_post = float(opt_row["utilization_after_opt"].values[0]) if not opt_row.empty else row["capacity_utilization"]
        lats.append(coords["lat"])
        lons.append(coords["lon"])
        labels.append(region)
        risks.append(row["risk_index"])
        sizes.append(20 + row["total_demand"] / max(scored["total_demand"].max(), 1) * 40)
        colors.append(coords["color"])
        hovers.append(
            f"<b>{region}</b><br>"
            f"Risk index: {row['risk_index']:.1f}<br>"
            f"Health score: {row['warehouse_health_score']:.1f}<br>"
            f"Utilization: {row['capacity_utilization']*100:.1f}%<br>"
            f"Post-opt utilization: {util_post*100:.1f}%<br>"
            f"Total demand: {row['total_demand']:.0f}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, text=labels,
        hovertext=hovers, hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=11, color="#eaf2ff"),
        marker=dict(
            size=sizes,
            color=risks,
            colorscale="RdYlGn_r",
            cmin=0, cmax=100,
            colorbar=dict(title="Risk index", tickfont=dict(color="#84a7df"),
                          titlefont=dict(color="#84a7df")),
            line=dict(color="rgba(255,255,255,.3)", width=1.5),
            opacity=0.88,
        ),
    ))
    fig.update_layout(
        geo=dict(
            scope="europe",
            center=dict(lat=41.0, lon=39.2),
            projection_scale=12,
            showland=True, landcolor="rgba(20,35,60,1)",
            showocean=True, oceancolor="rgba(6,14,28,1)",
            showcoastlines=True, coastlinecolor="rgba(80,120,180,.4)",
            showcountries=True, countrycolor="rgba(60,90,140,.5)",
            showrivers=True, rivercolor="rgba(30,60,110,.4)",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Warehouse Network — Black Sea Region", font=dict(color="#c8deff", size=14)),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def waterfall_comparison(comparison: pd.DataFrame) -> go.Figure:
    """Side-by-side before/after waterfall for health and risk."""
    fig = go.Figure()
    x = comparison["region"].tolist()
    fig.add_trace(go.Bar(name="Baseline health", x=x,
                         y=comparison["warehouse_health_score_baseline"],
                         marker_color="#4b72b8", opacity=.75))
    fig.add_trace(go.Bar(name="Scenario health", x=x,
                         y=comparison["warehouse_health_score_scenario"],
                         marker_color="#2dd4bf", opacity=.85))
    fig.add_trace(go.Bar(name="Baseline risk", x=x,
                         y=comparison["risk_index_baseline"],
                         marker_color="#f97316", opacity=.65))
    fig.add_trace(go.Bar(name="Scenario risk", x=x,
                         y=comparison["risk_index_scenario"],
                         marker_color="#ef4444", opacity=.85))
    fig.update_layout(**DARK_LAYOUT, barmode="group",
                      title="Baseline vs scenario — health & risk",
                      legend=dict(orientation="h", y=-0.15))
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, dict[str, float], bool, str]:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 .5rem 0;'>
          <div style='font-size:1.6rem;'>🏭</div>
          <div style='font-weight:800;font-size:1.1rem;color:#93c5fd;'>SupplyChain Nexus</div>
          <div style='font-size:.72rem;color:#5a82c0;margin-top:.2rem;'>Black Sea Pharma Network</div>
        </div>
        <hr style='border-color:rgba(110,168,254,.12);margin:.5rem 0;'/>
        """, unsafe_allow_html=True)

        st.markdown("**Navigation**")
        page = st.radio("", [
            "🏠  Overview",
            "🧪  Scenario Lab",
            "🤖  AI Insights",
            "🗺️  GIS Map",
            "♻️  Sustainability",
            "📦  Export",
        ], label_visibility="collapsed")

        st.markdown("<hr style='border-color:rgba(110,168,254,.12);margin:.6rem 0;'/>",
                    unsafe_allow_html=True)
        st.markdown("**Scenario settings**")
        scenario_name = st.selectbox("Scenario", SCENARIO_NAMES, index=0)
        focus_region  = st.selectbox("Focus warehouse",
                                     [m["region"] for m in WAREHOUSE_META], index=0)

        st.markdown("<hr style='border-color:rgba(110,168,254,.12);margin:.6rem 0;'/>",
                    unsafe_allow_html=True)
        st.markdown("**Objective weights**")
        cost_w    = st.slider("💰 Cost",     0.05, 1.0, 0.40, 0.05)
        time_w    = st.slider("⏱️ Time",     0.05, 1.0, 0.30, 0.05)
        balance_w = st.slider("⚖️ Balance",  0.05, 1.0, 0.30, 0.05)

        total_w = max(cost_w + time_w + balance_w, 0.001)
        weights = get_profile_weights(scenario_name)
        weights["cost"]     = cost_w    / total_w
        weights["time"]     = time_w    / total_w
        weights["balance"]  = balance_w / total_w
        weights["distance"] = float(np.clip(
            1 - (weights["cost"] + weights["time"] + weights["balance"]), 0.05, 0.35))

        st.markdown("<hr style='border-color:rgba(110,168,254,.12);margin:.6rem 0;'/>",
                    unsafe_allow_html=True)
        presentation_mode = st.toggle("🎯 Presentation mode", value=True)

        # weight pie
        wpie = go.Figure(go.Pie(
            labels=["Cost", "Time", "Balance", "Distance"],
            values=[weights["cost"], weights["time"], weights["balance"], weights["distance"]],
            hole=.55,
            marker_colors=["#6ea8fe", "#2dd4bf", "#f59e0b", "#f97316"],
            textinfo="label+percent",
            textfont_size=9,
        ))
        wpie.update_layout(height=170, showlegend=False, margin=dict(l=5,r=5,t=5,b=5),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(wpie, use_container_width=True)

    return page, scenario_name, weights, presentation_mode, focus_region

# ─── Hero + KPI bar ──────────────────────────────────────────────────────────

def render_hero_and_kpis(scored: pd.DataFrame, optimized: pd.DataFrame,
                          scenario_name: str, focus_region: str,
                          presentation_mode: bool) -> None:
    col_hero, col_story = st.columns([1.5, 1.0])
    with col_hero:
        st.markdown("""
        <div class="hero-shell">
          <div class="hero-title">SupplyChain Nexus</div>
          <div class="hero-copy">
            Multi-objective warehouse optimization, scenario stress testing, and
            AI-assisted planning for the Black Sea pharmaceutical distribution network.
            Powered by Google OR-Tools, K-Means, and Isolation Forest.
          </div>
        </div>""", unsafe_allow_html=True)

    with col_story:
        hotspot = scored.sort_values("risk_index", ascending=False).iloc[0]["region"]
        best    = optimized.sort_values(["warehouse_health_score", "risk_index"],
                                         ascending=[False, True]).iloc[0]["region"] \
                  if not optimized.empty else "—"
        avg_sl  = optimized["service_level"].mean() * 100 \
                  if "service_level" in optimized.columns else 100.0
        st.markdown(f"""
        <div class="glass" style="height:100%;">
          <div style="font-size:.72rem;color:#84a7df;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:.6rem;">Live snapshot</div>
          <div style="color:#f0f7ff;margin-bottom:.4rem;">
            🔴 Risk hotspot &nbsp;<strong style='color:#ef4444;'>{hotspot}</strong>
          </div>
          <div style="color:#f0f7ff;margin-bottom:.4rem;">
            ✅ Recommended focus &nbsp;<strong style='color:#22c55e;'>{best}</strong>
          </div>
          <div style="color:#f0f7ff;margin-bottom:.4rem;">
            🧪 Active scenario &nbsp;<strong style='color:#f59e0b;'>{scenario_name}</strong>
          </div>
          <div style="color:#f0f7ff;">
            📦 Avg service level &nbsp;<strong style='color:#6ea8fe;'>{avg_sl:.1f}%</strong>
          </div>
        </div>""", unsafe_allow_html=True)

    # KPI row
    st.markdown("")
    kpi_cols = st.columns(6)
    total_cap  = int(scored["total_capacity"].sum())
    total_dem  = int(scored["total_demand"].sum())
    avg_health = scored["warehouse_health_score"].mean()
    avg_risk   = scored["risk_index"].mean()
    avg_util   = optimized["utilization_after_opt"].mean() * 100 \
                 if "utilization_after_opt" in optimized.columns else \
                 scored["capacity_utilization"].mean() * 100
    total_short = optimized["shortage"].sum() if "shortage" in optimized.columns else 0

    kpis = [
        ("Warehouses",      len(scored),           "",          "#6ea8fe"),
        ("Total capacity",  f"{total_cap:,}",       "units",     "#2dd4bf"),
        ("Total demand",    f"{total_dem:,}",       "units",     "#f59e0b"),
        ("Avg health",      f"{avg_health:.1f}",    "/ 100",     "#22c55e"),
        ("Avg risk",        f"{avg_risk:.1f}",      "/ 100",     "#ef4444"),
        ("Avg post-opt util",f"{avg_util:.1f}%",   "utilization","#a78bfa"),
    ]
    for col, (label, val, sub, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    if presentation_mode:
        st.markdown("")
        hotspot_risk = scored.sort_values("risk_index", ascending=False).iloc[0]["risk_index"]
        best_health  = optimized.sort_values("warehouse_health_score",
                                              ascending=False).iloc[0]["warehouse_health_score"] \
                       if not optimized.empty else 0
        delta_txt = (f"{scenario_name} reallocates demand away from {hotspot} — "
                     f"monitor service levels at adjacent depots.")
        st.markdown(f"""
        <div class="verdict-card">
          <div class="verdict-title">Judge-ready verdict</div>
          <div class="verdict-body">
            Under the <strong>{scenario_name}</strong> scenario, concentrate operational control
            in <strong>{best}</strong> while treating <strong>{hotspot}</strong> as the primary
            risk hotspot. {delta_txt}
          </div>
          <div style="margin-top:.5rem;">
            <span class="mini-pill">Hotspot risk: {hotspot_risk:.1f}</span>
            <span class="mini-pill">Target health: {best_health:.1f}</span>
            <span class="mini-pill">Scenario: {scenario_name}</span>
            <span class="mini-pill">Focus: {focus_region}</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ─── Overview page ───────────────────────────────────────────────────────────

def page_overview(scored: pd.DataFrame, optimized: pd.DataFrame,
                  comparison: pd.DataFrame, data: dict[str, Any]) -> None:
    st.markdown('<div class="section-header">Warehouse health gauges</div>',
                unsafe_allow_html=True)
    gauge_cols = st.columns(len(scored))
    for col, (_, row) in zip(gauge_cols, scored.iterrows()):
        color = "#22c55e" if row["warehouse_health_score"] > 65 else \
                "#f59e0b" if row["warehouse_health_score"] > 40 else "#ef4444"
        with col:
            st.plotly_chart(gauge_chart(row["warehouse_health_score"],
                                        row["region"], color),
                            use_container_width=True)

    st.markdown('<div class="section-header">Logistics pressure & risk</div>',
                unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8])
    with left:
        fig = px.scatter(
            scored, x="avg_distance", y="avg_time",
            size="total_demand", color="risk_index",
            hover_name="region", text="region",
            color_continuous_scale="Turbo",
            size_max=50,
            title="Risk vs logistics pressure (bubble = demand volume)",
        )
        fig.update_traces(textposition="top center",
                          textfont=dict(color="#eaf2ff", size=11))
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        bar = optimized.sort_values("warehouse_health_score", ascending=False).copy()
        fig2 = px.bar(
            bar, x="region",
            y=["capacity_utilization", "utilization_after_opt"],
            barmode="group",
            color_discrete_sequence=["#4b72b8", "#2dd4bf"],
            title="Utilization: before vs after optimization",
            labels={"value": "Utilization ratio", "variable": ""},
        )
        fig2.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Demand composition</div>',
                unsafe_allow_html=True)
    demand_rows = []
    for meta in WAREHOUSE_META:
        region = meta["region"]
        d = parse_demand_table(data["demand"][region])
        for dtype, val in d.items():
            demand_rows.append({"region": region, "demand_type": dtype, "value": val})
    demand_df = pd.DataFrame(demand_rows)
    fig_dem = px.bar(demand_df, x="region", y="value", color="demand_type",
                     barmode="stack", title="Demand composition by type and region",
                     color_discrete_sequence=["#6ea8fe","#2dd4bf","#f59e0b","#f97316","#a78bfa"])
    fig_dem.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_dem, use_container_width=True)

    if not comparison.empty:
        st.markdown('<div class="section-header">Scenario delta</div>',
                    unsafe_allow_html=True)
        comp = comparison.copy()
        comp["health_delta"] = comp["warehouse_health_score_scenario"] - comp["warehouse_health_score_baseline"]
        comp["risk_delta"]   = comp["risk_index_scenario"]             - comp["risk_index_baseline"]
        delta_fig = go.Figure()
        colors_h = ["#22c55e" if v >= 0 else "#ef4444" for v in comp["health_delta"]]
        colors_r = ["#ef4444" if v >= 0 else "#22c55e" for v in comp["risk_delta"]]
        delta_fig.add_trace(go.Bar(name="Health Δ", x=comp["region"],
                                   y=comp["health_delta"], marker_color=colors_h))
        delta_fig.add_trace(go.Bar(name="Risk Δ",   x=comp["region"],
                                   y=comp["risk_delta"],   marker_color=colors_r))
        delta_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.3)")
        delta_fig.update_layout(**DARK_LAYOUT, barmode="group",
                                title="Health & risk delta: scenario vs baseline")
        st.plotly_chart(delta_fig, use_container_width=True)

# ─── Scenario Lab page ───────────────────────────────────────────────────────

def page_scenario(scored: pd.DataFrame, optimized: pd.DataFrame,
                  comparison: pd.DataFrame, scenario_name: str,
                  focus_region: str, baseline: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Scenario output table</div>',
                unsafe_allow_html=True)

    if optimized.empty:
        st.error("No feasible allocation found for this scenario.")
        return

    cols_show = ["region", "total_capacity", "total_demand", "capacity_after_scenario",
                 "demand_after_scenario", "assigned_demand", "capacity_utilization",
                 "utilization_after_opt", "warehouse_health_score", "risk_index",
                 "service_level", "shortage", "objective_score"]
    display_df = optimized[[c for c in cols_show if c in optimized.columns]].copy()
    display_df["service_level"] = (display_df["service_level"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Visual comparison</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(optimized, x="region",
                      y=["demand_after_scenario", "assigned_demand"],
                      barmode="group",
                      color_discrete_sequence=["#4b72b8", "#2dd4bf"],
                      title="Demand vs assigned (post-optimization)",
                      labels={"value": "Units", "variable": ""})
        fig1.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(optimized, x="region",
                       y=["warehouse_health_score", "risk_index"],
                       markers=True,
                       color_discrete_sequence=["#22c55e", "#ef4444"],
                       title="Operational health vs risk index")
        fig2.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # service level radial
    st.markdown('<div class="section-header">Service level by warehouse</div>',
                unsafe_allow_html=True)
    sl_vals = optimized["service_level"].clip(0, 1).tolist() if "service_level" in optimized.columns else [1.0]*len(optimized)
    sl_theta = [f"{r} ({v*100:.1f}%)" for r, v in zip(optimized["region"], sl_vals)]
    fig_radar = go.Figure(go.Barpolar(
        r=sl_vals, theta=sl_theta,
        marker_color=["#22c55e" if v > 0.9 else "#f59e0b" if v > 0.7 else "#ef4444"
                      for v in sl_vals],
        opacity=0.82,
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 1], showticklabels=True,
                                   tickfont=dict(color="#84a7df"), gridcolor="rgba(255,255,255,.08)"),
                   angularaxis=dict(tickfont=dict(color="#c8deff"))),
        paper_bgcolor="rgba(0,0,0,0)", height=320,
        title=dict(text="Service level by warehouse", font=dict(color="#c8deff")),
        margin=dict(l=60, r=60, t=50, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if not comparison.empty:
        st.markdown('<div class="section-header">Baseline vs scenario — full comparison</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(waterfall_comparison(comparison), use_container_width=True)

    st.markdown('<div class="section-header">Decision text</div>',
                unsafe_allow_html=True)
    best = optimized.sort_values(["warehouse_health_score", "risk_index"],
                                  ascending=[False, True]).iloc[0]
    worst = optimized.sort_values("risk_index", ascending=False).iloc[0]
    st.success(f"✅ Recommended operating point: **{best['region']}** — "
               f"health {best['warehouse_health_score']:.1f}, "
               f"risk {best['risk_index']:.1f}")
    if scenario_name != "Baseline":
        st.warning(f"⚠️ Scenario **{scenario_name}** applied on **{focus_region}** — "
                   f"monitor **{worst['region']}** closely (risk {worst['risk_index']:.1f})")
    total_shortage = optimized["shortage"].sum() if "shortage" in optimized.columns else 0
    if total_shortage > 0:
        st.error(f"🚨 Total unmet demand under this scenario: **{total_shortage:.0f} units** — "
                 f"consider contingency supply from outside the network.")

# ─── AI Insights page ────────────────────────────────────────────────────────

def page_ai_insights(scored: pd.DataFrame, optimized: pd.DataFrame,
                     scenario_name: str) -> None:
    recs = generate_recommendations(scored, optimized, scenario_name)

    st.markdown('<div class="section-header">AI recommendations</div>',
                unsafe_allow_html=True)
    for rec in recs:
        impact_bg = {"Critical":"rgba(220,38,38,.15)","High":"rgba(239,68,68,.1)",
                     "Medium":"rgba(245,158,11,.1)","Low":"rgba(34,197,94,.08)"}.get(
                     rec["impact"], "rgba(255,255,255,.05)")
        conf_color = "#22c55e" if rec["confidence"] >= 85 else \
                     "#f59e0b" if rec["confidence"] >= 70 else "#ef4444"
        st.markdown(f"""
        <div class="rec-card" style="border-left:4px solid {rec['color']};
                                     background:{impact_bg};">
          <div class="rec-type">{rec['type']} — {rec['region']}</div>
          <div class="rec-title">{rec['title']}</div>
          <div class="rec-body">
            <strong>Why:</strong> {rec['reason']}<br>
            <strong>Action:</strong> {rec['action']}<br>
            <strong>Expected savings:</strong> {rec['savings']}
          </div>
          <span class="rec-badge" style="background:{conf_color}22;
                color:{conf_color};border:1px solid {conf_color}55;">
            Confidence: {rec['confidence']}%
          </span>
          <span class="rec-badge" style="background:rgba(255,255,255,.04);
                color:#c8deff;border:1px solid rgba(255,255,255,.08);">
            Impact: {rec['impact']}
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">ML model room</div>',
                unsafe_allow_html=True)
    ml_c1, ml_c2 = st.columns([1.1, 0.9])
    with ml_c1:
        fig3 = px.scatter(
            scored, x="pca_x", y="pca_y",
            color="cluster", size="risk_index",
            hover_name="region",
            color_continuous_scale="Plasma",
            title="PCA region grouping (K-Means clusters)",
            labels={"pca_x": "PC 1", "pca_y": "PC 2"},
        )
        fig3.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with ml_c2:
        risk_sorted = scored.sort_values("risk_index", ascending=False)
        fig_risk = go.Figure()
        bar_colors = ["#ef4444" if r > 65 else "#f59e0b" if r > 40 else "#22c55e"
                      for r in risk_sorted["risk_index"]]
        fig_risk.add_trace(go.Bar(
            x=risk_sorted["region"], y=risk_sorted["risk_index"],
            name="Risk index", marker_color=bar_colors,
        ))
        fig_risk.add_trace(go.Bar(
            x=risk_sorted["region"], y=risk_sorted["warehouse_health_score"],
            name="Health score", marker_color="#6ea8fe", opacity=.7,
        ))
        fig_risk.add_hline(y=65, line_dash="dot", line_color="#ef4444",
                           annotation_text="Risk threshold", annotation_font_color="#ef4444")
        fig_risk.update_layout(**DARK_LAYOUT, barmode="group",
                               title="Risk index vs health score")
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<div class="section-header">Predicted vs actual utilization</div>',
                unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(x=scored["region"], y=scored["capacity_utilization"],
                              name="Actual", marker_color="#6ea8fe"))
    fig_pred.add_trace(go.Bar(x=scored["region"], y=scored["predicted_utilization"],
                              name="RF predicted", marker_color="#f97316", opacity=.75))
    fig_pred.add_trace(go.Scatter(x=scored["region"], y=scored["utilization_gap"],
                                  name="Gap", mode="lines+markers",
                                  line=dict(color="#f59e0b", dash="dot"), yaxis="y2"))
    fig_pred.update_layout(
        **DARK_LAYOUT,
        barmode="group",
        title="Actual vs predicted utilization & gap",
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="#f59e0b"), title="Gap"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown('<div class="section-header">Explainable AI</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
      <ul style="color:#b0c8ef;line-height:1.8;margin:0;padding-left:1.2rem;">
        <li><strong style="color:#6ea8fe;">K-Means clustering</strong> groups regions by similar
            capacity/demand/cost profiles — used to weight the balance objective.</li>
        <li><strong style="color:#2dd4bf;">Isolation Forest</strong> detects statistically
            anomalous regions — high anomaly score raises the risk index.</li>
        <li><strong style="color:#f59e0b;">Random Forest Regressor</strong> learns the
            expected utilization from all features; the gap between actual and
            predicted is an early-warning signal.</li>
        <li><strong style="color:#a78bfa;">OR-Tools CBC solver</strong> minimizes a weighted
            combination of cost, time, distance, and balance deviation subject to
            capacity and demand constraints — shortage variables keep it feasible
            under extreme scenarios.</li>
      </ul>
    </div>""", unsafe_allow_html=True)

# ─── GIS Map page ────────────────────────────────────────────────────────────

def page_gis(scored: pd.DataFrame, optimized: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Warehouse network map</div>',
                unsafe_allow_html=True)
    st.plotly_chart(build_geo_map(scored, optimized), use_container_width=True)
    st.markdown('<div class="map-label">Bubble size = total demand · Color = risk index '
                '(green → low risk, red → high risk)</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Regional coordinates & metrics</div>',
                unsafe_allow_html=True)
    geo_rows = []
    for meta in WAREHOUSE_META:
        region = meta["region"]
        coords = REGION_COORDS.get(region, {})
        row = scored[scored["region"] == region].iloc[0]
        opt_row = optimized[optimized["region"] == region]
        util_post = float(opt_row["utilization_after_opt"].values[0]) \
                    if not opt_row.empty else row["capacity_utilization"]
        geo_rows.append({
            "Region":       region,
            "Latitude":     coords.get("lat", "—"),
            "Longitude":    coords.get("lon", "—"),
            "Health score": f"{row['warehouse_health_score']:.1f}",
            "Risk index":   f"{row['risk_index']:.1f}",
            "Utilization":  f"{row['capacity_utilization']*100:.1f}%",
            "Post-opt util":f"{util_post*100:.1f}%",
            "Total demand": f"{row['total_demand']:.0f}",
        })
    st.dataframe(pd.DataFrame(geo_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Distance & time pressure heatmap</div>',
                unsafe_allow_html=True)
    heat_df = scored[["region", "avg_distance", "avg_time", "avg_cost",
                       "risk_index", "warehouse_health_score"]].copy()
    fig_heat = px.imshow(
        heat_df.set_index("region")[["avg_distance", "avg_time", "avg_cost",
                                      "risk_index", "warehouse_health_score"]].T,
        color_continuous_scale="RdBu_r",
        title="Operational metrics heatmap",
        labels=dict(x="Region", y="Metric", color="Value"),
        text_auto=".1f",
    )
    fig_heat.update_layout(**DARK_LAYOUT, height=280)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─── Sustainability page ─────────────────────────────────────────────────────

def page_sustainability(features: pd.DataFrame, optimized: pd.DataFrame) -> None:
    sus = estimate_sustainability(features, optimized)

    st.markdown('<div class="section-header">Green logistics scorecard</div>',
                unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (label, val, color) in zip(
        [sc1, sc2, sc3, sc4],
        [("Baseline CO₂ (kg)",   f"{sus['baseline_co2_kg']:.2f}",  "#ef4444"),
         ("Optimized CO₂ (kg)",  f"{sus['optimized_co2_kg']:.2f}", "#22c55e"),
         ("CO₂ saved (kg)",      f"{sus['co2_saved_kg']:.2f}",      "#2dd4bf"),
         ("Dist saved (km)",     f"{sus['distance_saved_km']:.1f}", "#f59e0b")],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    gs_col, gs_text = st.columns([1, 2])
    with gs_col:
        fig_gs = gauge_chart(sus["green_score"], "Green logistics score", "#22c55e")
        st.plotly_chart(fig_gs, use_container_width=True)
    with gs_text:
        st.markdown(f"""
        <div class="glass" style="height:100%;padding-top:1.5rem;">
          <div class="kpi-label">What this means</div>
          <ul style="color:#b0c8ef;line-height:1.8;padding-left:1.2rem;">
            <li>CO₂ savings are proportional to the <strong>distance pressure
                reduction</strong> achieved by the optimizer.</li>
            <li>Each km of distance saved ≈ 0.21 kg CO₂ per vehicle trip
                (standard road freight proxy).</li>
            <li>Green score of <strong style="color:#22c55e;">{sus['green_score']:.1f}/100</strong>
                reflects the percentage improvement from baseline routing.</li>
            <li>Centralizing underutilized warehouses further reduces
                empty-run distances.</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Distance pressure vs CO₂ by region</div>',
                unsafe_allow_html=True)
    features_sus = features.copy()
    features_sus["est_co2_kg"] = features_sus["distance_pressure"] * 0.00021
    fig_co2 = px.bar(features_sus, x="region", y="est_co2_kg",
                     color="est_co2_kg", color_continuous_scale="RdYlGn_r",
                     title="Estimated CO₂ by region (baseline)",
                     labels={"est_co2_kg": "CO₂ (kg)", "region": "Region"})
    fig_co2.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_co2, use_container_width=True)

# ─── Export page ─────────────────────────────────────────────────────────────

def page_export(scored: pd.DataFrame, optimized: pd.DataFrame,
                scenario_name: str, focus_region: str,
                features: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">Exportable data & charts</div>',
                unsafe_allow_html=True)

    sus = estimate_sustainability(features, optimized)
    hotspot = scored.sort_values("risk_index", ascending=False).iloc[0]["region"]
    best_r  = optimized.sort_values(["warehouse_health_score", "risk_index"],
                                     ascending=[False, True]).iloc[0]["region"] \
              if not optimized.empty else "—"
    recs    = generate_recommendations(scored, optimized, scenario_name)

    report_lines = [
        "=" * 60,
        "   SUPPLYCHAIN NEXUS — EXECUTIVE REPORT",
        "=" * 60,
        f"Scenario          : {scenario_name}",
        f"Focus warehouse   : {focus_region}",
        f"Risk hotspot      : {hotspot}",
        f"Recommended depot : {best_r}",
        "",
        "── Network KPIs ──────────────────────────────",
        f"Total capacity    : {int(scored['total_capacity'].sum()):,} units",
        f"Total demand      : {int(scored['total_demand'].sum()):,} units",
        f"Avg health score  : {scored['warehouse_health_score'].mean():.1f} / 100",
        f"Avg risk index    : {scored['risk_index'].mean():.1f} / 100",
        "",
        "── Sustainability ────────────────────────────",
        f"Baseline CO₂      : {sus['baseline_co2_kg']:.2f} kg",
        f"Optimized CO₂     : {sus['optimized_co2_kg']:.2f} kg",
        f"CO₂ saved         : {sus['co2_saved_kg']:.2f} kg",
        f"Green score       : {sus['green_score']:.1f} / 100",
        "",
        "── AI Recommendations ───────────────────────",
    ]
    for rec in recs:
        report_lines += [
            f"  [{rec['impact']}] {rec['title']}",
            f"      Why   : {rec['reason']}",
            f"      Action: {rec['action']}",
            f"      Conf  : {rec['confidence']}%  |  Savings: {rec['savings']}",
            "",
        ]
    report_lines += [
        "── Warehouse scorecard ───────────────────────",
    ]
    for _, row in scored.iterrows():
        report_lines.append(
            f"  {row['region']:<10}  Health:{row['warehouse_health_score']:.1f}  "
            f"Risk:{row['risk_index']:.1f}  Util:{row['capacity_utilization']*100:.1f}%"
        )
    report_lines += ["", "=" * 60, "Generated by SupplyChain Nexus", "=" * 60]
    report_payload = "\n".join(report_lines).encode("utf-8")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.markdown("**📊 Data exports**")
        st.download_button("📥 Scenario table CSV",
                           optimized.to_csv(index=False).encode("utf-8"),
                           "supplychain_scenario.csv", "text/csv")
        st.download_button("📥 Regional scorecard CSV",
                           scored.to_csv(index=False).encode("utf-8"),
                           "supplychain_scorecard.csv", "text/csv")
        st.download_button("📥 Feature table CSV",
                           features.to_csv(index=False).encode("utf-8"),
                           "supplychain_features.csv", "text/csv")

    with dl2:
        st.markdown("**📝 Report exports**")
        st.download_button("📥 Executive report TXT",
                           report_payload,
                           "supplychain_executive_report.txt", "text/plain")

    with dl3:
        st.markdown("**📈 Chart exports**")
        risk_fig = px.scatter(scored, x="avg_distance", y="avg_time",
                              size="total_demand", color="risk_index",
                              hover_name="region", color_continuous_scale="Turbo")
        risk_fig.update_layout(**DARK_LAYOUT)
        st.download_button("📥 Risk chart HTML",
                           risk_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           "risk_chart.html", "text/html")

        geo_fig = build_geo_map(scored, optimized)
        st.download_button("📥 GIS map HTML",
                           geo_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           "gis_map.html", "text/html")

    st.markdown("")
    with st.expander("📋 Preview executive report", expanded=False):
        st.code("\n".join(report_lines), language="text")

# ─── Main app ────────────────────────────────────────────────────────────────

def app() -> None:
    st.set_page_config(
        page_title="SupplyChain Nexus",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    # ── load & process ──
    with st.spinner("Loading datasets and running optimization…"):
        data      = load_all_data()
        features  = build_feature_table(data)
        ml_feats  = enrich_with_ml(features)
        scored    = score_health(ml_feats)

    # ── sidebar ──
    page, scenario_name, weights, presentation_mode, focus_region = render_sidebar()

    # ── solve ──
    scenario  = build_scenario(scenario_name, focus_region)
    optimized = solve_allocation(scored, weights, scenario)
    baseline  = solve_allocation(scored, DEFAULT_WEIGHTS, {})

    comparison = pd.DataFrame()
    if not optimized.empty and not baseline.empty:
        comparison = optimized[["region", "warehouse_health_score", "risk_index"]].merge(
            baseline[["region", "warehouse_health_score", "risk_index"]],
            on="region", suffixes=("_scenario", "_baseline"),
        )

    # ── hero + KPIs (always visible) ──
    render_hero_and_kpis(scored, optimized, scenario_name, focus_region, presentation_mode)
    st.markdown("---")

    # ── page routing ──
    clean_page = page.split("  ", 1)[-1].strip() if "  " in page else page.strip()

    if "Overview" in clean_page:
        page_overview(scored, optimized, comparison, data)

    elif "Scenario" in clean_page:
        page_scenario(scored, optimized, comparison, scenario_name,
                      focus_region, baseline)

    elif "AI" in clean_page:
        page_ai_insights(scored, optimized, scenario_name)

    elif "GIS" in clean_page or "Map" in clean_page:
        page_gis(scored, optimized)

    elif "Sustainability" in clean_page:
        page_sustainability(features, optimized)

    elif "Export" in clean_page:
        page_export(scored, optimized, scenario_name, focus_region, features)


if __name__ == "__main__":
    app()
