"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Dataset : Amazon Delivery Dataset (public dataset from Kaggle), 43,739 delivery records
File    : week2/data/raw/amazon_delivery.csv   (kept unchanged)

Steps
  1. Load and audit the raw data
  2. Hidden text problems (extra spaces, the text "NaN") and label standardisation
  3. Duplicate check
  4. Corrupted records and invalid values (age, rating)
  5. Missing values
  6. Dates and times (including pickups after midnight)
  7. GPS coordinates (sign errors and zero placeholders)
  8. Distance feature and outlier detection
  9. Feature engineering
 10. Encoding and normalisation
 11. Time coverage check (needed later for time series forecasting)
 12. Save cleaned data, a data quality report and figures

Run from the main folder of the repository:
    python week2/data_cleaning_pipeline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw" / "amazon_delivery.csv"
OUT = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.95)
log = {}
issues = []


def add_issue(name, column, rows, action):
    issues.append({"Issue": name, "Column(s)": column, "Rows affected": int(rows),
                   "Share of data %": round(rows / log["raw_rows"] * 100, 2), "Action": action})


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


raw = pd.read_csv(RAW)
df = raw.copy()
log["raw_rows"], log["raw_cols"] = raw.shape
log["dtypes"] = {c: str(t) for c, t in raw.dtypes.items()}
log["missing_raw"] = {c: int(v) for c, v in raw.isnull().sum().items() if v > 0}
log["unique_order_ids"] = int(raw["Order_ID"].nunique())
log["date_min"], log["date_max"] = raw["Order_Date"].min(), raw["Order_Date"].max()
log["raw_describe"] = raw[["Agent_Age", "Agent_Rating", "Delivery_Time"]].describe().round(2).to_dict()
print(raw.info())
print(raw.describe().round(2))

raw_dist = haversine(raw.Store_Latitude, raw.Store_Longitude, raw.Drop_Latitude, raw.Drop_Longitude)
log["raw_distance"] = {"mean": round(float(raw_dist.mean()), 1), "median": round(float(raw_dist.median()), 1),
                       "max": round(float(raw_dist.max()), 1), "over_50km": int((raw_dist > 50).sum())}

text_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
ws = {c: int((raw[c].notna() & (raw[c] != raw[c].str.strip())).sum()) for c in text_cols}
log["whitespace_rows"] = ws
log["traffic_low_match_before"] = int((raw["Traffic"] == "Low").sum())
ws_any = (raw[text_cols].apply(lambda s: s.notna() & (s != s.str.strip()))).any(axis=1)
add_issue("Extra spaces at the end of text labels", "Traffic, Vehicle, Area", ws_any.sum(),
          "Strip whitespace from all text columns")

for c in text_cols + ["Order_Time"]:
    df[c] = df[c].str.strip()
nan_text = {c: int((df[c] == "NaN").sum()) for c in ["Traffic", "Order_Time"]}
log["nan_text"] = nan_text
add_issue('Missing values stored as the text "NaN"', "Traffic, Order_Time", nan_text["Traffic"],
          "Convert to real missing values")
df = df.replace("NaN", np.nan)

for c in ["Weather", "Vehicle", "Area", "Category", "Traffic"]:
    df[c] = df[c].str.title()
spell = int((df["Area"] == "Metropolitian").sum())
df["Area"] = df["Area"].replace({"Metropolitian": "Metropolitan"})
add_issue("Misspelt label 'Metropolitian'", "Area", spell, "Correct the spelling to 'Metropolitan'")
log["traffic_low_match_after"] = int((df["Traffic"] == "Low").sum())
log["labels"] = {c: sorted(df[c].dropna().unique().tolist()) for c in ["Weather", "Traffic", "Vehicle", "Area"]}

log["dup_full"] = int(df.duplicated().sum())
log["dup_id"] = int(df["Order_ID"].duplicated().sum())
log["dup_ignoring_id"] = int(df.drop(columns="Order_ID").duplicated().sum())

bad_age = df["Agent_Age"] < 18
bad_rating = df["Agent_Rating"] > 5
low_rating_one = df["Agent_Rating"] == 1.0
corrupt = df["Weather"].isna() & df["Traffic"].isna() & df["Order_Time"].isna()
log["age_under_18"] = int(bad_age.sum())
log["rating_over_5"] = int(bad_rating.sum())
log["rating_equal_1"] = int(low_rating_one.sum())
log["corrupt_records"] = int(corrupt.sum())
log["age_under_18_inside_corrupt"] = int((bad_age & corrupt).sum())
log["rating_over_5_inside_corrupt"] = int((bad_rating & corrupt).sum())
log["rating_1_inside_corrupt"] = int((low_rating_one & corrupt).sum())
log["corrupt_age_values"] = df.loc[corrupt, "Agent_Age"].value_counts().to_dict()
add_issue("Corrupted records (weather, traffic and order time all missing)",
          "Weather, Traffic, Order_Time", corrupt.sum(), "Remove the records")
add_issue("Impossible agent age (15, below working age)", "Agent_Age", bad_age.sum(), "Removed with the corrupted records")
add_issue("Impossible rating (6.0 on a 5-point scale)", "Agent_Rating", bad_rating.sum(), "Removed with the corrupted records")
df = df[~corrupt].copy()
log["after_corrupt_removed"] = len(df)
log["age_range_after"] = [int(df["Agent_Age"].min()), int(df["Agent_Age"].max())]
log["rating_range_after"] = [float(df["Agent_Rating"].min()), float(df["Agent_Rating"].max())]

log["missing_before_imputation"] = {c: int(v) for c, v in df.isnull().sum().items() if v > 0}
add_issue("Missing agent rating", "Agent_Rating", raw["Agent_Rating"].isna().sum(), "Fill with the median rating")
log["rating_mean_median_skew"] = [round(df["Agent_Rating"].mean(), 3), float(df["Agent_Rating"].median()),
                                  round(df["Agent_Rating"].skew(), 2)]
median_rating = df["Agent_Rating"].median()
n_rating_missing = int(df["Agent_Rating"].isna().sum())
df["Agent_Rating"] = df["Agent_Rating"].fillna(median_rating)
log["rating_imputed"] = n_rating_missing

df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%Y-%m-%d")
df["order_dt"] = pd.to_datetime(df["Order_Date"].dt.strftime("%Y-%m-%d") + " " + df["Order_Time"])
df["pickup_dt"] = pd.to_datetime(df["Order_Date"].dt.strftime("%Y-%m-%d") + " " + df["Pickup_Time"])
prep_raw = (df["pickup_dt"] - df["order_dt"]).dt.total_seconds() / 60
midnight = prep_raw < 0
log["prep_raw"] = {"mean": round(float(prep_raw.mean()), 1), "min": float(prep_raw.min()),
                   "negative_rows": int(midnight.sum())}
log["midnight_order_hours"] = sorted(df.loc[midnight, "order_dt"].dt.hour.unique().tolist())
log["midnight_pickup_hours"] = sorted(df.loc[midnight, "pickup_dt"].dt.hour.unique().tolist())
add_issue("Pickup time earlier than order time (pickup after midnight)", "Pickup_Time", raw_prep_neg := midnight.sum(),
          "Add one day to the pickup (not an error, order at 23:xx and pickup at 00:xx)")
df.loc[midnight, "pickup_dt"] += pd.Timedelta(days=1)
df["prep_minutes"] = (df["pickup_dt"] - df["order_dt"]).dt.total_seconds() / 60
log["prep_clean"] = {"min": float(df["prep_minutes"].min()), "max": float(df["prep_minutes"].max()),
                     "mean": round(float(df["prep_minutes"].mean()), 2),
                     "values": sorted(df["prep_minutes"].unique().tolist())}

neg = (df["Store_Latitude"] < 0) | (df["Store_Longitude"] < 0)
zero = (df["Store_Latitude"] == 0) | (df["Store_Longitude"] == 0)
log["store_negative_rows"] = int(neg.sum())
log["store_negative_lat"] = int((df["Store_Latitude"] < 0).sum())
log["store_negative_lon"] = int((df["Store_Longitude"] < 0).sum())
log["store_zero_rows"] = int(zero.sum())
log["drop_lat_min_in_zero_rows"] = round(float(df.loc[zero, "Drop_Latitude"].min()), 3)
d_before = haversine(df.loc[neg, "Store_Latitude"], df.loc[neg, "Store_Longitude"],
                     df.loc[neg, "Drop_Latitude"], df.loc[neg, "Drop_Longitude"])
d_fixed = haversine(df.loc[neg, "Store_Latitude"].abs(), df.loc[neg, "Store_Longitude"].abs(),
                    df.loc[neg, "Drop_Latitude"], df.loc[neg, "Drop_Longitude"])
log["neg_distance_before_median_km"] = round(float(d_before.median()), 0)
log["neg_distance_after_median_km"] = round(float(d_fixed.median()), 1)
log["neg_distance_after_max_km"] = round(float(d_fixed.max()), 1)
dz = haversine(df.loc[zero, "Store_Latitude"], df.loc[zero, "Store_Longitude"],
               df.loc[zero, "Drop_Latitude"], df.loc[zero, "Drop_Longitude"])
log["zero_rows_distance_median_km"] = round(float(dz.median()), 1)
log["zero_rows_distance_max_km"] = round(float(dz.max()), 1)
log["delivery_mean_zero_rows"] = round(float(df.loc[zero, "Delivery_Time"].mean()), 1)
log["delivery_mean_other_rows"] = round(float(df.loc[~zero, "Delivery_Time"].mean()), 1)
add_issue("Negative store coordinates (sign error)", "Store_Latitude, Store_Longitude", neg.sum(),
          "Repair: take the absolute value (checked against drop location)")
add_issue("Store coordinates equal to 0 (placeholder)", "Store_Latitude, Store_Longitude", zero.sum(),
          "Remove the records (cannot be repaired)")
before_coords = df[["Store_Latitude", "Store_Longitude"]].copy()
df.loc[neg, ["Store_Latitude", "Store_Longitude"]] = df.loc[neg, ["Store_Latitude", "Store_Longitude"]].abs()
df = df[~(df["Store_Latitude"] == 0) & ~(df["Store_Longitude"] == 0)].copy()
log["after_zero_removed"] = len(df)
log["lat_range"] = [round(df[["Store_Latitude", "Drop_Latitude"]].min().min(), 2), round(df[["Store_Latitude", "Drop_Latitude"]].max().max(), 2)]
log["lon_range"] = [round(df[["Store_Longitude", "Drop_Longitude"]].min().min(), 2), round(df[["Store_Longitude", "Drop_Longitude"]].max().max(), 2)]

df["distance_km"] = haversine(df["Store_Latitude"], df["Store_Longitude"],
                              df["Drop_Latitude"], df["Drop_Longitude"]).round(2)
q1, q3 = df["distance_km"].quantile([0.25, 0.75])
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
log["distance"] = {"mean": round(float(df["distance_km"].mean()), 2), "median": round(float(df["distance_km"].median()), 2),
                   "min": float(df["distance_km"].min()), "max": float(df["distance_km"].max()),
                   "iqr_upper": round(float(hi), 2), "iqr_outliers": int((df["distance_km"] > hi).sum())}

dt = df["Delivery_Time"]
q1, q3 = dt.quantile([0.25, 0.75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
z = (dt - dt.mean()) / dt.std()
iqr_out = dt > hi
log["delivery_time"] = {"mean": round(float(dt.mean()), 1), "median": float(dt.median()), "std": round(float(dt.std()), 1),
                        "min": int(dt.min()), "max": int(dt.max()), "skew": round(float(dt.skew()), 2),
                        "iqr_lower": float(lo), "iqr_upper": float(hi), "iqr_outliers": int(iqr_out.sum()),
                        "zscore_outliers": int((z.abs() > 3).sum()), "outlier_values": sorted(dt[iqr_out].unique().tolist())}
log["iqr_outlier_area"] = df.loc[iqr_out, "Area"].value_counts().to_dict()
log["iqr_outlier_traffic"] = df.loc[iqr_out, "Traffic"].value_counts().to_dict()
df["long_delivery_flag"] = iqr_out.astype(int)
add_issue("Long delivery times flagged by the IQR rule (above 265 min)", "Delivery_Time", iqr_out.sum(),
          "Keep (valid values, target variable), add a flag column")

df["hour"] = df["order_dt"].dt.hour
df["day_of_week"] = df["order_dt"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
hourly = df["hour"].value_counts().sort_index()
peak_hours = hourly[hourly > hourly.mean()].index.tolist()
df["is_peak_hour"] = df["hour"].isin(peak_hours).astype(int)
df["part_of_day"] = pd.cut(df["hour"], bins=[-1, 4, 11, 16, 23], labels=["Night", "Morning", "Afternoon", "Evening"])
log["peak_hours"] = peak_hours
log["hourly_orders"] = {int(k): int(v) for k, v in hourly.items()}
log["weekend_share"] = round(float(df["is_weekend"].mean() * 100), 1)

df = df.drop(columns=["order_dt", "pickup_dt", "Order_Time", "Pickup_Time"])
log["clean_rows"], log["clean_cols"] = df.shape
log["missing_after"] = int(df.isnull().sum().sum())
log["rows_removed"] = log["raw_rows"] - log["clean_rows"]
log["rows_removed_pct"] = round(log["rows_removed"] / log["raw_rows"] * 100, 2)
df.to_csv(OUT / "cleaned_delivery_data.csv", index=False)

m = df.copy()
m["Traffic_code"] = m["Traffic"].map({"Low": 0, "Medium": 1, "High": 2, "Jam": 3})
m = pd.get_dummies(m, columns=["Weather", "Vehicle", "Area", "Category", "part_of_day"], drop_first=True)
num_cols = ["Agent_Age", "Agent_Rating", "distance_km", "prep_minutes"]
mm = MinMaxScaler()
m[[c + "_scaled" for c in num_cols]] = mm.fit_transform(m[num_cols])
m["distance_z"] = StandardScaler().fit_transform(m[["distance_km"]])
model_ready = m.drop(columns=["Order_ID", "Order_Date", "Traffic", "Store_Latitude", "Store_Longitude",
                              "Drop_Latitude", "Drop_Longitude"])
model_ready.to_csv(OUT / "model_ready_data.csv", index=False)
log["model_ready_shape"] = list(model_ready.shape)
log["vehicle_counts"] = df["Vehicle"].value_counts().to_dict()
log["area_counts"] = df["Area"].value_counts().to_dict()

daily = df.groupby("Order_Date").size().rename("orders")
all_days = pd.date_range(daily.index.min(), daily.index.max())
missing_days = all_days.difference(daily.index)
daily_full = daily.reindex(all_days)
daily_full.rename_axis("date").to_csv(OUT / "daily_orders.csv")
log["time"] = {"first_day": str(all_days[0].date()), "last_day": str(all_days[-1].date()),
               "calendar_days": len(all_days), "days_with_orders": int(len(daily)),
               "missing_days": [str(d.date()) for d in missing_days],
               "avg_orders_per_day": round(float(daily.mean()), 1), "min_day": int(daily.min()), "max_day": int(daily.max())}
log["missing_days_raw"] = int(len(pd.date_range(pd.to_datetime(raw["Order_Date"]).min(), pd.to_datetime(raw["Order_Date"]).max())) - raw["Order_Date"].nunique())

dd = daily.copy(); dd.index = pd.to_datetime(dd.index)
log["orders_by_weekday"] = dd.groupby(dd.index.dayofweek).mean().round(0).astype(int).to_dict()
log["orders_first_week_avg"] = round(float(dd[dd.index <= "2022-02-18"].mean()), 0)
log["orders_after_gap_avg"] = round(float(dd[dd.index >= "2022-03-01"].mean()), 0)
log["distinct_store_locations"] = int(df[["Store_Latitude", "Store_Longitude"]].round(1).drop_duplicates().shape[0])
log["distinct_distance_values"] = int(df["distance_km"].round(1).nunique())
log["delivery_by_traffic"] = df.groupby("Traffic")["Delivery_Time"].mean().round(1).reindex(["Low", "Medium", "High", "Jam"]).to_dict()
log["delivery_by_area"] = df.groupby("Area")["Delivery_Time"].mean().round(1).to_dict()
log["delivery_by_vehicle"] = df.groupby("Vehicle")["Delivery_Time"].mean().round(1).to_dict()
log["orders_in_dead_hours"] = int(df["hour"].between(1, 7).sum())
log["raw_delivery_by_traffic_dirty_match"] = int((raw["Traffic"] == "Jam").sum())

pd.DataFrame(issues).to_csv(OUT / "data_quality_report.csv", index=False)
log["issues"] = issues

iss = pd.DataFrame(issues).sort_values("Rows affected")
iss = iss[iss["Issue"].str.startswith(("Extra", "Missing values stored", "Corrupted", "Negative", "Store", "Pickup", "Missing agent", "Impossible"))]
fig, ax = plt.subplots(figsize=(8.5, 4.0))
short = {"Extra spaces at the end of text labels": "Trailing spaces in text",
         'Missing values stored as the text "NaN"': 'Missing values stored as "NaN"',
         "Corrupted records (weather, traffic and order time all missing)": "Corrupted records",
         "Impossible agent age (15, below working age)": "Impossible age (15)",
         "Impossible rating (6.0 on a 5-point scale)": "Impossible rating (6.0)",
         "Missing agent rating": "Missing rating",
         "Negative store coordinates (sign error)": "Negative store coordinates",
         "Store coordinates equal to 0 (placeholder)": "Zero store coordinates",
         "Pickup time earlier than order time (pickup after midnight)": "Pickup after midnight"}
iss["label"] = iss["Issue"].map(short)
ax.barh(iss["label"], iss["Rows affected"], color="#C0504D")
ax.set_xscale("log"); ax.set_xlabel("Rows affected (log scale)")
for i, v in enumerate(iss["Rows affected"]): ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlim(10, iss["Rows affected"].max() * 6)
ax.set_title("Data Quality Issues Found in the Raw Dataset")
fig.tight_layout(); fig.savefig(FIG / "01_data_quality_issues.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
axes[0].scatter(raw["Store_Longitude"], raw["Store_Latitude"], s=3, alpha=0.3, color="#C0504D")
axes[0].set_title("Raw store coordinates"); axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].annotate("(0, 0) placeholders", xy=(0, 0), xytext=(-45, 12), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1))
axes[0].annotate("negative latitude\n(sign error)", xy=(-76, -16), xytext=(-85, 8), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1))
axes[1].scatter(df["Store_Longitude"], df["Store_Latitude"], s=3, alpha=0.3, color="#4F81BD")
axes[1].set_title("Cleaned store coordinates"); axes[1].set_xlabel("Longitude")
fig.tight_layout(); fig.savefig(FIG / "02_store_coordinates_before_after.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
sns.boxplot(y=raw_dist, ax=axes[0], color="#E6A5A3"); axes[0].set_yscale("log")
axes[0].set_title("Raw distance (log scale)"); axes[0].set_ylabel("km")
sns.histplot(df["distance_km"], bins=40, ax=axes[1], color="#4F81BD")
axes[1].set_title("Cleaned distance"); axes[1].set_xlabel("Distance (km)")
fig.tight_layout(); fig.savefig(FIG / "03_distance_before_after.png", dpi=130); plt.close(fig)

raw_prep = (pd.to_datetime(raw["Pickup_Time"], format="%H:%M:%S") - pd.to_datetime(raw["Order_Time"].str.strip(), format="%H:%M:%S", errors="coerce")).dt.total_seconds() / 60
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
vc = raw_prep.dropna().astype(int).value_counts().sort_index()
axes[0].bar([str(i) for i in vc.index], vc.values, color=["#C0504D" if i < 0 else "#B7C7E0" for i in vc.index])
axes[0].set_title("Raw pickup minus order time"); axes[0].set_xlabel("Minutes"); axes[0].set_ylabel("Orders")
vc2 = df["prep_minutes"].astype(int).value_counts().sort_index()
axes[1].bar([str(i) for i in vc2.index], vc2.values, color="#9BBB59")
axes[1].set_title("After midnight correction"); axes[1].set_xlabel("Minutes")
fig.tight_layout(); fig.savefig(FIG / "04_prep_time_before_after.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [2, 1.2]})
sns.histplot(df["Delivery_Time"], bins=40, kde=True, ax=axes[0], color="#4F81BD")
axes[0].axvline(log["delivery_time"]["iqr_upper"], color="red", ls="--", label=f"IQR upper limit {log['delivery_time']['iqr_upper']:.0f}")
axes[0].set_title("Delivery time (cleaned data)"); axes[0].set_xlabel("Minutes"); axes[0].legend(fontsize=8)
sns.boxplot(x="Traffic", y="Delivery_Time", hue="Traffic", data=df, order=["Low", "Medium", "High", "Jam"],
            palette=dict(zip(["Low", "Medium", "High", "Jam"], sns.color_palette("YlOrRd", 4))),
            hue_order=["Low", "Medium", "High", "Jam"], legend=False, ax=axes[1])
axes[1].set_title("By traffic level"); axes[1].set_ylabel("")
fig.tight_layout(); fig.savefig(FIG / "05_delivery_time_outliers.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
sns.histplot(df["distance_km"], bins=30, ax=axes[0], color="#4F81BD"); axes[0].set_title("Original (km)")
sns.histplot(model_ready["distance_km_scaled"], bins=30, ax=axes[1], color="#8064A2"); axes[1].set_title("Min-Max (0 to 1)")
sns.histplot(model_ready["distance_z"], bins=30, ax=axes[2], color="#F79646"); axes[2].set_title("Z-score")
for a in axes: a.set_xlabel(""); a.set_ylabel("Count")
fig.tight_layout(); fig.savefig(FIG / "06_scaling_comparison.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
axes[0].plot(daily_full.index, daily_full.values, marker="o", ms=3, color="#1F3864")
for d in missing_days: axes[0].axvline(d, color="#C0504D", alpha=0.25, lw=3)
axes[0].set_title("Orders per day (red bands = days with no data)"); axes[0].set_ylabel("Orders")
fig.autofmt_xdate()
cols = ["#C0504D" if h in peak_hours else "#B7C7E0" for h in hourly.index]
axes[1].bar(hourly.index, hourly.values, color=cols); axes[1].set_title("Orders by hour (red = peak)")
axes[1].set_xlabel("Hour of day")
fig.tight_layout(); fig.savefig(FIG / "07_time_coverage.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.4))
corr_cols = ["Delivery_Time", "distance_km", "prep_minutes", "Agent_Rating", "Agent_Age", "hour"]
tmp = df[corr_cols].copy(); tmp["Traffic_code"] = df["Traffic"].map({"Low": 0, "Medium": 1, "High": 2, "Jam": 3})
cm = tmp.corr()
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation After Cleaning")
fig.tight_layout(); fig.savefig(FIG / "08_correlation_after_cleaning.png", dpi=130); plt.close(fig)
log["corr_delivery"] = cm["Delivery_Time"].round(2).to_dict()

with open(BASE / "cleaning_summary.json", "w") as f:
    json.dump(log, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(log, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
