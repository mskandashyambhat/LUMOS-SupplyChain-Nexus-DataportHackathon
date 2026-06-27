# 🏭 SupplyChain Nexus

> **An intelligent supply chain risk analytics and decision support platform for the Black Sea pharmaceutical warehouse network.**
> Built for the Dataport Hackathon. Combines multi-objective optimization, machine learning, scenario simulation, geospatial visualization, and AI-powered recommendations — all from raw Excel datasets.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Live demo — how to run it](#2-live-demo--how-to-run-it)
3. [Project structure](#3-project-structure)
4. [Understanding the data](#4-understanding-the-data)
5. [How the app works — end-to-end pipeline](#5-how-the-app-works--end-to-end-pipeline)
6. [Page-by-page guide](#6-page-by-page-guide)
7. [Controls & settings reference](#7-controls--settings-reference)
8. [AI & ML models explained](#8-ai--ml-models-explained)
9. [Optimization engine explained](#9-optimization-engine-explained)
10. [Scenario simulations explained](#10-scenario-simulations-explained)
11. [Sustainability module explained](#11-sustainability-module-explained)
12. [Exports reference](#12-exports-reference)
13. [Tech stack](#13-tech-stack)
14. [Installation & requirements](#14-installation--requirements)

---

## 1. What is this project?

Modern pharmaceutical supply chains in the Black Sea region (Turkey) need to balance warehouse capacity, transportation cost, delivery time, and regional demand — while minimizing operational risk.

**SupplyChain Nexus** solves this by turning raw Excel workbooks into a fully interactive decision cockpit. It:

- Calculates how stressed each warehouse is right now
- Runs a **linear programming optimizer** (Google OR-Tools) to find the best demand allocation across warehouses
- Uses **machine learning** (K-Means, Isolation Forest, RandomForest) to flag risks and predict future utilization
- Lets you **simulate disruptions** — outages, capacity cuts, demand spikes — and instantly see the impact
- Generates **AI-powered recommendations** with confidence scores, expected savings, and action plans
- Shows everything on an **interactive geospatial map** of the Black Sea coastline
- Tracks **CO₂ / sustainability impact** of the optimized routing vs. baseline
- Exports all results as CSVs, HTML charts, and a formatted executive report

---

## 2. Live demo — how to run it

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/mskandashyambhat/LUMOS-SupplyChain-Nexus-DataportHackathon.git
cd LUMOS-SupplyChain-Nexus-DataportHackathon
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser automatically.

> **Note:** The first load takes ~5–10 seconds while the optimizer solves all allocations. Subsequent interactions are instant because results are cached.

---

## 3. Project structure

```
LUMOS-SupplyChain-Nexus-DataportHackathon/
│
├── app.py                          # Main application — all logic, UI, pages
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── .streamlit/
│   └── config.toml                 # Dark theme configuration for Streamlit
│
├── Centralization of Pharmaceutical Warehouses.../
│   ├── CapacityClustered.xlsx      # Warehouse capacity per cluster per region
│   ├── DemandCluster.xlsx          # Customer demand per cluster per region
│   ├── CostCluster.xlsx            # Transportation cost matrix
│   ├── DistanceCluster.xlsx        # Distance matrix (km)
│   ├── Time.xlsx                   # Travel time matrix (hours)
│   ├── CostsMWC-Clustered.xlsx     # Multi-warehouse cost data
│   └── GeoLocations.xlsx           # Latitude/longitude for warehouses and customers
│
└── Others/
    ├── Hackathon.pdf / .txt        # Original hackathon problem statement
    └── idea.pdf / .txt             # Initial feature design document
```

---

## 4. Understanding the data

All source data lives in the `Centralization of Pharmaceutical Warehouses...` folder. There are **6 Excel workbooks**, each with one sheet per region.

### Regions covered

| Region  | Latitude  | Longitude | Color code |
|---------|-----------|-----------|------------|
| Rize    | 41.0201 N | 40.5234 E | Blue       |
| Trabzon | 41.0015 N | 39.7178 E | Teal       |
| Giresun | 40.9128 N | 38.3895 E | Amber      |
| Ordu    | 40.9862 N | 37.8797 E | Orange     |

All four are coastal cities on Turkey's Black Sea coast, each with a pharmaceutical warehouse cluster.

---

### CapacityClustered.xlsx

**What it contains:** How much each warehouse (cluster point) can physically store.

**Sheet names:** `Capacity Rize`, `Capacity Trabzon`, `Capacity Giresun`, `Capacity Ordu`

**Structure:**
```
P=1                     ← cluster label (P=1, P=2, P=3 ...)
capacity    val1  val2  ← total storage capacity across time points
capacityCold  ...       ← cold-chain specific capacity
capacityNorm  ...       ← normal temperature capacity
capCritCold   ...       ← critical cold storage
capCritNorm   ...       ← critical normal storage
P=2
...
```

**How to read it:** Each block starting with `P=N` defines a warehouse cluster. The row labelled `capacity` gives the total capacity values across columns (time periods or sub-locations). The app sums all capacity values per cluster to get `total_capacity` per region.

---

### DemandCluster.xlsx

**What it contains:** How much demand (pharmaceutical units) each customer cluster generates.

**Sheet names:** `Demand Rize`, `Demand Trabzon`, `Demand Giresun`, `Demand Ordu`
**Additional sheet:** `Gercek Data` — actual (real) observed demand for validation

**Structure:**
```
demand        val1  val2  val3 ...   ← total demand
demandCold    val1  val2  ...        ← cold-chain demand
demandNorm    val1  val2  ...        ← normal demand
demandColdCrit  ...                  ← critical cold demand
demandNormCrit  ...                  ← critical normal demand
```

**How to read it:** Each row starting with `demand` (or `demandCold`, etc.) contains demand values across customer points. The app sums all values per demand type and computes `total_demand`, `demand_cold`, and `demand_norm` per region.

---

### CostCluster.xlsx

**What it contains:** Transportation cost matrix — how much it costs to move goods between origin and destination cluster points.

**Sheet names:** `Cost Rize`, `Cost Trabzon`, `Cost Giresun`, `Cost Ordu`

**Structure:**
```
         Dest1   Dest2   Dest3 ...
Origin1  cost    cost    cost
Origin2  cost    cost    cost
...
```

**How to read it:** A standard origin-destination cost matrix. Row 1 is a header (skipped by the app). The app melts this into long format (origin, target, cost) and takes the mean as `avg_cost` per region.

---

### DistanceCluster.xlsx

**What it contains:** Distance (km) between cluster origin and destination points.

**Sheet names:** `Distance Rize`, `Distance Trabzon`, `Distance Giresun`, `Distance Ordu`

**Structure:** Same origin-destination matrix format as CostCluster. The app computes `avg_distance` per region.

---

### Time.xlsx

**What it contains:** Travel time (hours) between cluster origin and destination points.

**Sheet names:** `Time Rize`, `Time Trabzon`, `Time Giresun`, `Time Ordu`

**Structure:** Same origin-destination matrix format. The app computes `avg_time` per region.

---

### CostsMWC-Clustered.xlsx

**What it contains:** Multi-warehouse cost data (MWC) for clustered scenarios — costs when multiple warehouses serve the same cluster zone.

**Sheet names:** `Costs Rize`, `Costs Trabzon`, `Costs Giresun`, `Costs Ordu`

---

### GeoLocations.xlsx

**What it contains:** Latitude and longitude coordinates for warehouses and customer points.

**Sheet names:** One sheet per region (`Rize`, `Trabzon`, `Giresun`, `Ordu`) + `P-WH` (warehouse anchor points)

**Structure:**
```
lat      lon
41.020   40.523
...
```

The app uses `P-WH` sheet for warehouse map anchors and per-region sheets for customer scatter points.

---

## 5. How the app works — end-to-end pipeline

```
Excel workbooks (6 files)
        │
        ▼
load_all_data()           ← reads all sheets into memory, cached
        │
        ▼
build_feature_table()     ← aggregates to one row per region:
                            total_capacity, total_demand, capacity_utilization,
                            avg_cost, avg_distance, avg_time,
                            demand_cold, demand_norm,
                            cost_pressure, distance_pressure, time_pressure
        │
        ▼
enrich_with_ml()          ← adds ML columns:
                            cluster (K-Means), ml_risk_flag (Isolation Forest),
                            pca_x / pca_y (PCA 2D), predicted_utilization
                            (RandomForest), utilization_gap
        │
        ▼
score_health()            ← computes:
                            warehouse_health_score (0–100)
                            risk_index (0–100)
        │
        ▼
solve_allocation()        ← OR-Tools CBC linear program:
                            inputs: demand, capacity, cost, distance, time
                            outputs: assigned_demand, utilization_after_opt,
                                     service_level, shortage, objective_score
        │
        ├── baseline solve  (DEFAULT_WEIGHTS, no scenario)
        └── scenario solve  (user weights + active scenario modifier)
                            comparison = scenario vs baseline
        │
        ▼
render_sidebar()          ← user controls: page, scenario, weights, region
        │
        ▼
Page router               ← dispatches to one of 6 pages based on sidebar nav
```

---

## 6. Page-by-page guide

### 🏠 Overview

The main executive dashboard. Always start here.

**What you see:**
- **Hero banner** with a live snapshot panel (hotspot, recommendation, active scenario, avg service level)
- **6 KPI cards** — Warehouses · Total capacity · Total demand · Avg health score · Avg risk index · Avg post-opt utilization
- **Judge-ready verdict card** (visible in Presentation mode) — a one-paragraph narrative summary ready to read out loud
- **4 health gauges** — one per warehouse, color-coded (green > 65, amber 40–65, red < 40)
- **Risk vs logistics pressure scatter** — bubbles sized by demand, colored by risk index
- **Before vs after optimization bar chart** — utilization before and after the solver runs
- **Demand composition stacked bar** — cold, normal, critical cold, critical normal demand by region
- **Scenario delta chart** — health and risk change vs baseline (green = improved, red = worsened)

---

### 🧪 Scenario Lab

Where you stress-test the network.

**What you see:**
- **Full scenario output table** — every metric per warehouse including service level and shortage units
- **Demand vs assigned bar chart** — how much demand was requested vs how much was actually assigned
- **Health vs risk line chart** — operational health score vs risk index per warehouse
- **Service level polar chart** — radial view of % demand fulfilled (green > 90%, amber > 70%, red below)
- **Baseline vs scenario waterfall** — side-by-side comparison of health and risk before/after scenario
- **Decision text** — auto-generated recommendation text + shortage warning if any units are unmet

**How to use it:**
1. Select a scenario in the sidebar (e.g. "Warehouse outage")
2. Select which warehouse is affected in "Focus warehouse"
3. The page recalculates and shows the full before/after comparison instantly

---

### 🤖 AI Insights

The intelligence layer — recommendations, ML visualization, and explainability.

**What you see:**
- **AI recommendation cards** — dynamically generated based on actual utilization, risk scores, and shortage data. Each card shows:
  - Alert type (Overload / Underutilization / High Risk / Shortage / Optimal)
  - Region name and reason
  - Recommended action
  - Confidence percentage
  - Expected savings / impact
  - Impact level (Critical / High / Medium / Low)
- **PCA region grouping scatter** — 2D view of how K-Means clusters the 4 regions
- **Risk index vs health score bar** — side-by-side with a red threshold line at 65
- **Predicted vs actual utilization chart** — Random Forest prediction overlaid on actual, with gap line
- **Explainable AI panel** — plain-English explanation of what each model does and why

---

### 🗺️ GIS Map

The geospatial view of the network.

**What you see:**
- **Interactive Plotly map** centered on the Black Sea coastline
  - Each warehouse is a bubble: **size = total demand volume**, **color = risk index** (green → red gradient)
  - Hover over any bubble to see: region name, risk index, health score, utilization %, post-opt utilization %, total demand
- **Coordinates & metrics table** — lat/lon, health, risk, utilization, post-opt utilization, demand for all 4 regions
- **Operational metrics heatmap** — color-coded grid of avg_distance, avg_time, avg_cost, risk_index, health_score across all regions

---

### ♻️ Sustainability

The green logistics dashboard.

**What you see:**
- **4 sustainability KPI cards** — Baseline CO₂ · Optimized CO₂ · CO₂ saved · Distance saved (km)
- **Green logistics score gauge** (0–100)
- **Explanation panel** — how the CO₂ estimate is calculated and what it means
- **Regional CO₂ bar chart** — estimated emissions by warehouse region, colored by intensity

---

### 📦 Export

Download everything for your presentation.

**What you see:**
- **Data exports** — Scenario table CSV · Regional scorecard CSV · Feature table CSV
- **Report exports** — Formatted executive report TXT with all KPIs, AI recommendations, and sustainability data
- **Chart exports** — Risk chart HTML · GIS map HTML (fully interactive, shareable as standalone files)
- **Report preview** — expandable inline preview of the executive report

---

## 7. Controls & settings reference

All controls are in the left sidebar.

| Control | What it does |
|---|---|
| **Navigation radio** | Switches between the 6 pages |
| **Scenario dropdown** | Selects which disruption scenario to simulate |
| **Focus warehouse dropdown** | Which warehouse the outage scenario targets |
| **💰 Cost slider** | Increases optimizer weight on minimizing transport cost |
| **⏱️ Time slider** | Increases optimizer weight on minimizing travel time |
| **⚖️ Balance slider** | Increases optimizer weight on balancing utilization evenly |
| **Weight pie chart** | Live visualization of how the 4 weights (cost/time/balance/distance) are distributed |
| **🎯 Presentation mode toggle** | Shows/hides the judge-ready verdict card and presentation narrative |

**How weights work:**
The three sliders (cost, time, balance) are normalized to sum to 1. Distance weight is automatically computed as the remainder, clamped to 5%–35%. Changing scenario also auto-adjusts default weights (e.g., "Warehouse outage" prioritizes balance heavily).

---

## 8. AI & ML models explained

### K-Means Clustering
- **Input features:** total_capacity, total_demand, capacity_utilization, avg_cost, avg_distance, avg_time
- **Output:** `cluster` label (0, 1, or 2)
- **Purpose:** Groups regions with similar operational profiles. Used as a feature in the health score formula and visualized in the PCA scatter plot.
- **Parameters:** k=3 (or less if fewer regions), n_init=10, random_state=42

### Isolation Forest
- **Input:** Same 6 features
- **Output:** `ml_risk_flag` — 1 if anomalous, -1 if normal (negated so 1 = flagged)
- **Purpose:** Detects regions that are statistically unusual compared to the group. A flagged region raises its risk index and lowers its health score.
- **Parameters:** contamination=10–30% depending on dataset size

### PCA (Principal Component Analysis)
- **Input:** Standardized 6-feature matrix
- **Output:** `pca_x`, `pca_y` — 2D coordinates
- **Purpose:** Reduces dimensionality for visualization. The PCA scatter in AI Insights shows how similar/different the 4 regions are in feature space.

### Random Forest Regressor
- **Input:** Same 6 features
- **Target:** `capacity_utilization`
- **Output:** `predicted_utilization` and `utilization_gap`
- **Purpose:** Learns the "expected" utilization from the data. The gap between actual and predicted is an early-warning signal — if a region's actual utilization is much higher than predicted, it's being over-stressed.
- **Parameters:** 200 trees, StandardScaler preprocessing, random_state=42

### Health score formula
```
warehouse_health_score = 100
    - 55 × capacity_utilization   (high utilization → lower health)
    - 15 × ml_risk_flag           (anomaly flag → lower health)
    + 8  × (1 - cluster/max_cluster)  (cluster diversity bonus)
clipped to [0, 100]
```

### Risk index formula
```
risk_index = 40 × capacity_utilization
           + 30 × ml_risk_flag
           + 15 × (avg_distance / max_avg_distance)
           + 15 × (avg_time / max_avg_time)
clipped to [0, 100]
```

> **Risk threshold:** 65 — regions above this are flagged as critical and trigger AI recommendations.

---

## 9. Optimization engine explained

The core of the platform is a **multi-objective linear program** solved by Google OR-Tools CBC solver.

### Decision variables
- `x[i, j]` — units of demand from region i assigned to warehouse j (continuous, ≥ 0)
- `shortage[i]` — unmet demand units for region i (continuous, ≥ 0, penalized heavily)

### Constraints
1. **Demand coverage:** All demand for region i must be assigned or marked as shortage
   `Σⱼ x[i,j] + shortage[i] = demand[i]`
2. **Capacity limit:** Each warehouse j cannot be assigned more than its capacity
   `Σᵢ x[i,j] ≤ capacity[j]`

### Objective (minimize)
```
Σᵢ Σⱼ  x[i,j] × (w_cost × cost[j]
                 + w_distance × distance[j]
                 + w_time × time[j]
                 + w_balance × |capacity[j]/total_demand - 1|)

+ Σᵢ shortage[i] × 1000    ← shortage penalty

+ Σⱼ utilization_deviation[j] × 25 × w_balance    ← balance term

+ Σⱼ allocation_deviation[j] × 0.5 × w_balance    ← proportional allocation term
```

### Output columns
| Column | Meaning |
|---|---|
| `assigned_demand` | Units of demand assigned to this warehouse by the solver |
| `optimization_share` | Fraction of total network demand assigned here |
| `utilization_after_opt` | assigned_demand / capacity (post-optimization fill rate) |
| `service_level` | % of regional demand met (1 = 100%, shortage = < 1) |
| `shortage` | Units of demand left unmet (should be 0 under normal scenarios) |
| `objective_score` | Normalized objective value — lower is better |

---

## 10. Scenario simulations explained

Select a scenario in the sidebar to see how the network responds.

| Scenario | What changes | Auto weight adjustment |
|---|---|---|
| **Baseline** | Nothing — normal operation | Cost 40%, Time 30%, Distance 20%, Balance 10% |
| **Warehouse outage** | Selected warehouse capacity → 0 | Balance weight raised to 40% |
| **Capacity stress** | All warehouse capacities × 0.75 | Balance raised to 30% |
| **Demand spike** | All regional demand × 1.25 | Time weight raised to 35% |

The solver re-runs instantly with the modified inputs. The Scenario Lab page shows:
- How assigned demand shifted between warehouses
- Which warehouse now has the highest risk
- Whether any demand goes unmet (shortage > 0)
- How health and risk scores changed vs. baseline

**How to run a stress test step-by-step:**
1. Go to sidebar → select **"Warehouse outage"**
2. Set **Focus warehouse** to "Ordu"
3. Navigate to **🧪 Scenario Lab**
4. Check the service level polar chart — any red sections mean unmet demand
5. Check the decision text at the bottom for the recommendation
6. Navigate to **🤖 AI Insights** to see if shortage recommendations appeared

---

## 11. Sustainability module explained

The sustainability estimates are **proxies based on distance pressure**, not actual GPS route data.

**Formula:**
```
baseline_co2_kg = sum(distance_pressure across all regions) × 0.00021
                  where distance_pressure = total_demand × avg_distance

optimized_co2_kg = baseline_co2_kg × utilization_efficiency_factor

co2_saved_kg = baseline_co2_kg - optimized_co2_kg

green_score = min(100, 40 + (1 - utilization_efficiency_factor) × 600)
```

**The 0.00021 factor** is a standard road freight proxy: ~0.21 kg CO₂ per tonne-km, scaled to the dataset's unit magnitude.

The **green score** reflects how much the optimizer reduced routing inefficiency relative to baseline — a score above 60 indicates meaningful sustainability improvement.

---

## 12. Exports reference

All downloads are on the **📦 Export** page.

| File | Format | Contents |
|---|---|---|
| `supplychain_scenario.csv` | CSV | Full scenario output table with all metrics |
| `supplychain_scorecard.csv` | CSV | Regional health/risk/utilization scores |
| `supplychain_features.csv` | CSV | Raw feature table (capacity, demand, cost, distance, time, pressures) |
| `supplychain_executive_report.txt` | TXT | Formatted report with KPIs, AI recommendations, sustainability data |
| `risk_chart.html` | HTML | Standalone interactive risk scatter chart (Plotly) |
| `gis_map.html` | HTML | Standalone interactive geospatial map (Plotly) |

---

## 13. Tech stack

| Layer | Library | Purpose |
|---|---|---|
| Web app | Streamlit 1.50 | UI framework, page routing, caching |
| Optimization | Google OR-Tools (ortools) | CBC linear programming solver |
| ML — clustering | scikit-learn KMeans | Region segmentation |
| ML — anomaly detection | scikit-learn IsolationForest | Risk flagging |
| ML — prediction | scikit-learn RandomForestRegressor | Utilization forecasting |
| ML — visualization | scikit-learn PCA | Dimensionality reduction for scatter |
| Data processing | Pandas, NumPy | Data loading, feature engineering |
| Excel reading | openpyxl (via Pandas) | Reading .xlsx workbooks |
| Visualization | Plotly Express + Graph Objects | All interactive charts and maps |
| Geospatial | Plotly Scattergeo | Warehouse network map |
| Styling | Custom CSS injected via st.markdown | Dark theme, KPI cards, gauges |

---

## 14. Installation & requirements

```
pandas
numpy
openpyxl
scikit-learn
streamlit
plotly
ortools
scipy
```

Install with:
```bash
pip install -r requirements.txt
```

Minimum Python version: **3.10**

Tested on: Python 3.12, macOS (Apple Silicon), Streamlit 1.50, Plotly 6.8

---

## About

Built by **LUMOS** for the **Dataport Hackathon**.

Problem domain: Centralization of Pharmaceutical Warehouses — Black Sea Region, Turkey.

The platform transforms raw logistics Excel data into optimized warehouse allocations, scenario-based risk planning, explainable AI recommendations, and interactive geospatial visualizations — enabling organizations to make faster, data-driven decisions that reduce cost, improve utilization, and strengthen supply chain resilience.
