# Last-Mile Delivery Analysis – Logistics Data Analyst Internship

This repository contains my work for the Logistics Data Analyst internship (September – October 2026). The project looks at how data analysis and Python can be used to improve last-mile delivery performance for a regional e-commerce courier.

## Scenario
A courier company delivers parcels across a large city region. It faces late deliveries, uneven rider workload, failed first attempts and rising cost per parcel. The project uses data to find the causes, predict delays and suggest better planning.

## KPIs Used
- On-Time Delivery Rate
- Average Delivery Time
- Cost per Delivery
- First Attempt Success Rate
- Fleet Utilisation

## Repository Structure
```
├── week1/    Strategic planning and data exploration
├── week2/    Data collection, cleaning and preprocessing   (coming)
├── week3/    Data analysis and visualisation               (coming)
├── week4/    Predictive modelling and optimisation         (coming)
├── requirements.txt
└── README.md
```

## Week 1 – Strategic Planning
`week1/week1_code_snippets.py` shows the planned approach: loading data, cleaning, KPI calculation, exploratory plots, a random forest delivery-time model and K-Means zone clustering. It uses a small simulated dataset so the script runs without any download. The full written plan is in the Week 1 report (Word document).

## How to Run
```
pip install -r requirements.txt
python week1/week1_code_snippets.py
```
Run the command from the main folder of the repository. The script prints the KPIs and model results and saves three charts in the `week1` folder.

## Tools
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn

## Note
The data in Week 1 is simulated for practice. Later weeks will use a public logistics dataset.
