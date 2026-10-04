# Last-Mile Delivery Analysis – Logistics Data Analyst Internship

This repository contains my work for the Logistics Data Analyst internship (September – October 2026). The project uses Python and data analysis to study last-mile delivery performance for an e-commerce courier, using a real public dataset.

## Dataset
**Amazon Delivery Dataset** (Kaggle, public): 43,739 delivery records from India (11 Feb – 6 Apr 2022) with agent details, store and drop-off GPS coordinates, order and pickup times, weather, traffic, vehicle, area, product category and delivery time. The original file is kept unchanged in `week2/data/raw/`.

## Scenario and KPIs
A courier delivers e-commerce parcels across Indian cities. The project looks at late deliveries, the effect of traffic, weather and distance, and how planning can be improved.
- On-Time Delivery Rate
- Average Delivery Time
- Cost per Delivery
- First Attempt Success Rate
- Fleet Utilisation

## Repository Structure
```
├── week1/    Strategic planning and data exploration
├── week2/    Data collection, cleaning and preprocessing (real data)
├── week3/    Exploratory analysis, statistics, time series and visualisation
├── week4/    Predictive modelling, forecasting, optimisation (coming)
├── requirements.txt
└── README.md
```

## Week 1 – Strategic Planning
`week1/week1_code_snippets.py` shows the planned approach (data loading, cleaning, KPIs, exploratory plots, a delivery-time model and zone clustering). It runs on a small simulated dataset because it is a planning illustration. The written plan is in the Week 1 report.

## Week 2 – Data Collection, Cleaning and Preprocessing
`week2/data_cleaning_pipeline.py` audits and cleans the real dataset step by step and records every change.

**Problems found and handled (counts are rows):**

| Problem | Rows | Treatment |
|---------|------|-----------|
| Extra spaces in Traffic, Vehicle, Area labels | 43,739 | Strip spaces |
| Missing values stored as the text "NaN" | 91 | Convert to real missing values |
| Misspelt label "Metropolitian" | 32,698 | Correct spelling |
| Corrupted records (weather, traffic, order time missing; age 15, ratings 1.0 or 6.0) | 91 | Remove |
| Missing agent rating | 54 | Fill with median (4.7) |
| Pickup after midnight shown as negative preparation time | 828 | Repair (add one day) |
| Negative store latitude (sign error) | 156 | Repair (absolute value) |
| Store coordinates equal to 0 | 3,495 | Remove |
| Delivery time above IQR limit (all exactly 270 min) | 74 | Keep and flag |
| Duplicates | 0 | None needed |

Result: 40,153 clean rows (91.8% kept), no missing values, average delivery time unchanged (124.9 → 125.1 min), average distance corrected from 38.6 km to 9.7 km.

| File | Purpose |
|------|---------|
| `week2/data_cleaning_pipeline.py` | Full cleaning pipeline |
| `week2/data/raw/amazon_delivery.csv` | Original data (unchanged) |
| `week2/data/cleaned_delivery_data.csv` | Cleaned data with new features |
| `week2/data/model_ready_data.csv` | Encoded and scaled data for modelling |
| `week2/data/daily_orders.csv` | Orders per day (used for time series work) |
| `week2/data/data_quality_report.csv` | List of issues, rows affected and actions |
| `week2/cleaning_summary.json` | All counts and statistics used in the report |
| `week2/figures/` | Eight charts |

Time check for forecasting: 55 calendar days, 44 with data (11 missing days: 19–28 Feb and 22 Mar); evening hours 17:00–23:00 hold most of the orders.

## Week 3 – Advanced Data Analysis and Visualisation
`week3/eda_and_visualization.py` analyses the 40,153 cleaned deliveries from Week 2: descriptive statistics, statistical tests (ANOVA, Kruskal-Wallis, eta-squared), correlations, a multivariable regression, time series analysis (daily and hourly volume, autocorrelation, decomposition), bottleneck and cost analysis, and 15 charts.

**Main findings**
- Grocery orders take 26.5 min on average against 131.5 min for all other categories (a separate fast service).
- Traffic level is almost fixed by the hour of day. Delivery time rises from 101.4 min (low traffic) to 147.9 min (jam); on-time rate falls from 92.4% to 52.7% (target 150 min, assumed).
- Days alternate strictly between two types (22 days each): All-day days (about 984 orders, 110.7 min) and Evening-only days (about 841 orders, 142.0 min). The 2-day cycle explains 98.8% of the variation in daily orders; there is no weekly pattern.
- Evening load is 119 orders per hour on Evening-only days against 70 on All-day days; deliveries in the same hours are 22 to 33 min slower.
- Motorcycles are 14.5 min slower than scooters at every traffic level; agents rated 4.0 or lower take 180.5 min against about 120.
- Delays are not tied to particular stores (67 store areas, average time only 116.8–136.1 min).
- A multivariable regression explains 63.3% of the variation in delivery time (baseline for Week 4).

**Assumptions:** the dataset has no cost data and no promised delivery time, so cost (rider time Rs 1.5/min + vehicle running cost per km) and the 150-minute target are my own assumptions and are labelled as such in the report.

| File | Purpose |
|------|---------|
| `week3/eda_and_visualization.py` | Full analysis and chart script |
| `week3/Week3_Analysis_Visualization.docx` | Week 3 report |
| `week3/eda_summary.json` | All numbers used in the report |
| `week3/data/analysis_dataset.csv` | Cleaned data plus assumed cost and on-time columns |
| `week3/data/daily_summary.csv`, `hourly_summary.csv` | Time series tables (daily and hourly) |
| `week3/data/summary_by_*.csv`, `store_area_summary.csv`, `descriptive_statistics.csv` | Group and descriptive tables |
| `week3/figures/` | 15 charts |

## How to Run
```
pip install -r requirements.txt
python week2/data_cleaning_pipeline.py
python week3/eda_and_visualization.py
```
Run from the main folder of the repository. Week 2 reads the raw file and writes the cleaned data files and charts. Week 3 reads `week2/data/cleaned_delivery_data.csv`, so run Week 2 first.

## Tools
Python, pandas, NumPy, SciPy, statsmodels, matplotlib, seaborn, scikit-learn

## Note
Week 1 contains a small simulated example for planning. From Week 2 onwards all analysis uses the real public dataset.
