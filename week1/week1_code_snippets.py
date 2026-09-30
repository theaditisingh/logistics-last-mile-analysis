import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

np.random.seed(42)


def make_sample_data(n=2000):
    order_time = pd.Timestamp("2026-09-01 08:00") + pd.to_timedelta(
        np.random.randint(0, 30 * 12 * 60, n), unit="m")

    distance = np.round(np.random.gamma(2.5, 2.5, n), 1)
    load = np.round(np.random.uniform(0.5, 25, n), 1)
    stops = np.random.randint(1, 15, n)
    traffic = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
    weather = np.random.choice(["Sunny", "Cloudy", "Rainy", "Fog"], n, p=[0.5, 0.25, 0.15, 0.1])
    area = np.random.choice(["Urban", "Semi-Urban", "Metropolitan"], n)

    traffic_effect = pd.Series(traffic).map({"Low": 0, "Medium": 8, "High": 18}).values
    weather_effect = pd.Series(weather).map({"Sunny": 0, "Cloudy": 2, "Rainy": 10, "Fog": 7}).values
    minutes = 12 + 3.2 * distance + 0.4 * load + 1.5 * stops \
              + traffic_effect + weather_effect + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "order_time": order_time,
        "distance_km": distance,
        "load_kg": load,
        "stops": stops,
        "traffic": traffic,
        "weather": weather,
        "area": area,
        "latitude": 28.40 + np.random.rand(n) * 0.35,
        "longitude": 76.95 + np.random.rand(n) * 0.45,
        "promised_minutes": 60,
        "total_cost": np.round(20 + 4.5 * distance + np.random.normal(0, 6, n), 2),
    })
    df["delivered_time"] = df["order_time"] + pd.to_timedelta(minutes, unit="m")

    df.loc[np.random.choice(n, 40, replace=False), "weather"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "delivered_time"] = pd.NaT
    df = pd.concat([df, df.sample(15, random_state=1)])
    return df


df = make_sample_data()

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

df = df.drop_duplicates(subset="order_id")

df["order_time"] = pd.to_datetime(df["order_time"])
df["delivered_time"] = pd.to_datetime(df["delivered_time"])
df["delivery_minutes"] = (df["delivered_time"] - df["order_time"]).dt.total_seconds() / 60

df["weather"] = df["weather"].fillna(df["weather"].mode()[0])
df = df.dropna(subset=["delivery_minutes"])

q1, q3 = df["delivery_minutes"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
before = len(df)
df = df[(df["delivery_minutes"] >= low) & (df["delivery_minutes"] <= high)]
print(f"\nOutliers removed: {before - len(df)}")

df["hour"] = df["order_time"].dt.hour
df["day_of_week"] = df["order_time"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["on_time"] = df["delivery_minutes"] <= df["promised_minutes"]

otd_rate = df["on_time"].mean() * 100
avg_time = df["delivery_minutes"].mean()
cost_per_delivery = df["total_cost"].sum() / len(df)

print("\n--- KPIs ---")
print(f"On-time delivery rate : {otd_rate:.1f}%")
print(f"Average delivery time : {avg_time:.1f} minutes")
print(f"Cost per delivery     : Rs {cost_per_delivery:.2f}")

kpi_area = df.groupby("area").agg(
    otd=("on_time", "mean"),
    avg_minutes=("delivery_minutes", "mean"),
    cost=("total_cost", "mean"),
    parcels=("order_id", "count"),
).sort_values("otd")
print("\nKPIs by area:\n", kpi_area)

sns.histplot(df["delivery_minutes"], bins=40, kde=True)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (minutes)")
plt.savefig("delivery_time_distribution.png", dpi=120, bbox_inches="tight")
plt.clf()

sns.boxplot(x="traffic", y="delivery_minutes", data=df, order=["Low", "Medium", "High"])
plt.title("Delivery Time by Traffic Level")
plt.savefig("delivery_time_by_traffic.png", dpi=120, bbox_inches="tight")
plt.clf()

sns.heatmap(df[["delivery_minutes", "distance_km", "load_kg", "stops"]].corr(),
            annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.clf()

features = ["distance_km", "hour", "is_weekend", "load_kg", "stops",
            "traffic", "weather", "area"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["delivery_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n--- Model results ---")
print("MAE  :", round(mean_absolute_error(y_test, pred), 2))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("R2   :", round(r2_score(y_test, pred), 3))

cv = cross_val_score(model, X, y, cv=5, scoring="r2")
print("5-fold CV R2:", round(cv.mean(), 3))

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features:\n", importance.head(5))

km = KMeans(n_clusters=8, random_state=42, n_init=10)
df["zone"] = km.fit_predict(df[["latitude", "longitude"]])
print("\nParcels per zone:\n", df["zone"].value_counts().sort_index())
