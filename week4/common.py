import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent

SRC = BASE / "data" / "analysis_dataset.csv"
DAILY = BASE / "data" / "daily_summary.csv"

FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

(BASE / "data").mkdir(exist_ok=True)
SEED = 42
SLA = 150                                  # assumed delivery target (minutes), same as Week 3
TRAFFIC = ["Low", "Medium", "High", "Jam"]
CUTOFF = "2022-03-25"                      # temporal split: train up to this date, test after it

NUM = ["distance_km", "Traffic_code", "is_grocery", "Agent_Age", "Agent_Rating", "hour", "prep_minutes", "is_weekend"]
CAT = ["Weather", "Vehicle", "Area"]
FEATURES_A = NUM + CAT
FEATURES_B = NUM + ["day_type_evening"] + CAT          # adds the day type (alternates every day, so it can be planned)
TARGET = "Delivery_Time"


def load():
    df = pd.read_csv(SRC, parse_dates=["Order_Date"])
    daily = pd.read_csv(DAILY, parse_dates=["date"]).set_index("date")
    df["day_type"] = df["Order_Date"].map(daily["day_type"])          # All-day / Evening-only (from Week 3)
    df["day_type_evening"] = (df["day_type"] == "Evening-only").astype(int)
    return df


def split_random(df, test_size=0.2):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    return df.iloc[idx[n_test:]].copy(), df.iloc[idx[:n_test]].copy()


def split_time(df):
    return df[df["Order_Date"] <= CUTOFF].copy(), df[df["Order_Date"] > CUTOFF].copy()


def preprocessor(features):
    num = [c for c in features if c not in CAT]
    cat = [c for c in features if c in CAT]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])


def pipe(model, features):
    return Pipeline([("prep", preprocessor(features)), ("model", model)])


def candidates(features):
    return {
        "Linear Regression": pipe(LinearRegression(), features),
        "Ridge Regression": pipe(Ridge(alpha=1.0), features),
        "Decision Tree": pipe(DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=SEED), features),
        "Random Forest": pipe(RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=10,
                                                    max_features=0.6, random_state=SEED, n_jobs=1), features),
        "Gradient Boosting": pipe(HistGradientBoostingRegressor(random_state=SEED), features),
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    if hasattr(o, "item"):
        return clean(o.item())
    return o


def dump(obj, name):
    with open(BASE / name, "w") as f:
        json.dump(clean(obj), f, indent=1)
