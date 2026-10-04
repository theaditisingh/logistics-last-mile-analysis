import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "analysis_dataset.csv"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
(BASE / "data").mkdir(exist_ok=True)
SLA = 150
OUT = {}
TRAFFIC = ["Low", "Medium", "High", "Jam"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def r(x, n=1):
    return round(float(x), n)


df = pd.read_csv(SRC, parse_dates=["Order_Date"])
df["Traffic"] = pd.Categorical(df["Traffic"], categories=TRAFFIC, ordered=True)
df["Traffic_code"] = df["Traffic"].map({"Low": 0, "Medium": 1, "High": 2, "Jam": 3}).astype(int)
df["on_time"] = (df["Delivery_Time"] <= SLA).astype(int)
df["is_grocery"] = (df["Category"] == "Grocery").astype(int)

RIDER_RS_PER_MIN = 1.5
RUN_RS_PER_KM = {"Motorcycle": 3.0, "Scooter": 2.5, "Van": 8.0}
df["cost_rider"] = RIDER_RS_PER_MIN * df["Delivery_Time"]
df["cost_vehicle"] = df["Vehicle"].map(RUN_RS_PER_KM) * df["distance_km"]
df["est_cost"] = df["cost_rider"] + df["cost_vehicle"]
df.to_csv(BASE / "data" / "analysis_dataset.csv", index=False)

n = len(df)
OUT["rows"] = n
OUT["days"] = int(df["Order_Date"].nunique())
OUT["sla"] = SLA
OUT["sla_sensitivity"] = {str(s): r((df["Delivery_Time"] <= s).mean() * 100) for s in (120, 150, 180)}
OUT["kpi"] = {"on_time_rate": r(df["on_time"].mean() * 100), "avg_delivery_time": r(df["Delivery_Time"].mean()),
              "median_delivery_time": r(df["Delivery_Time"].median()),
              "avg_est_cost": r(df["est_cost"].mean()), "orders_per_day": r(n / OUT["days"]),
              "rider_hours_total": int(df["Delivery_Time"].sum() / 60)}
OUT["cost_share_rider_pct"] = r(df["cost_rider"].sum() / df["est_cost"].sum() * 100)

cols = ["Delivery_Time", "distance_km", "prep_minutes", "Agent_Age", "Agent_Rating", "est_cost"]
desc = df[cols].agg(["mean", "median", "std", "min", "max", "skew"]).T
desc["mode"] = [df[c].round(0).mode()[0] for c in cols]
desc["cv_pct"] = desc["std"] / desc["mean"] * 100
desc = desc.round(2)
desc.to_csv(BASE / "data" / "descriptive_statistics.csv")
OUT["describe"] = desc.to_dict("index")
OUT["kurtosis_delivery"] = r(df["Delivery_Time"].kurt(), 2)

gro, non = df[df["is_grocery"] == 1], df[df["is_grocery"] == 0]
OUT["grocery"] = {"orders": len(gro), "share_pct": r(len(gro) / n * 100), "mean": r(gro["Delivery_Time"].mean()),
                  "median": r(gro["Delivery_Time"].median()), "max": int(gro["Delivery_Time"].max()),
                  "on_time_pct": r(gro["on_time"].mean() * 100)}
OUT["non_grocery"] = {"orders": len(non), "mean": r(non["Delivery_Time"].mean()), "median": r(non["Delivery_Time"].median()),
                      "std": r(non["Delivery_Time"].std()), "on_time_pct": r(non["on_time"].mean() * 100)}
cat = df.groupby("Category")["Delivery_Time"].agg(["count", "mean"]).round(1).sort_values("mean")
OUT["category_mean_range_non_grocery"] = [r(cat.drop("Grocery")["mean"].min()), r(cat.drop("Grocery")["mean"].max())]
OUT["category_volume_range"] = [int(cat["count"].min()), int(cat["count"].max())]


def group_table(col):
    g = df.groupby(col, observed=True).agg(
        orders=("Order_ID", "count"), avg_time=("Delivery_Time", "mean"), median_time=("Delivery_Time", "median"),
        on_time_pct=("on_time", lambda s: s.mean() * 100), avg_cost=("est_cost", "mean"),
        avg_distance=("distance_km", "mean")).round(1)
    return g


for col in ["Traffic", "Weather", "Vehicle", "Area", "part_of_day"]:
    t = group_table(col)
    t.to_csv(BASE / "data" / f"summary_by_{col.lower()}.csv")
    OUT[f"by_{col.lower()}"] = t.to_dict("index")
OUT["by_traffic"] = {str(k): v for k, v in OUT["by_traffic"].items()}


def effect(col, data=df):
    groups = [g["Delivery_Time"].values for _, g in data.groupby(col, observed=True)]
    f, p_anova = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    gm = data["Delivery_Time"].mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_t = ((data["Delivery_Time"] - gm) ** 2).sum()
    return {"eta_sq": round(float(ss_b / ss_t), 3), "F": r(f, 0), "p_anova": float(p_anova), "p_kruskal": float(p_kw)}


OUT["effects_all"] = {c: effect(c) for c in ["Category", "Traffic", "hour", "Weather", "Area", "part_of_day", "Vehicle", "is_weekend"]}
OUT["effects_non_grocery"] = {c: effect(c, non) for c in ["Traffic", "Weather", "Area", "Vehicle", "Category"]}

w = df.groupby("Weather")["Delivery_Time"].agg(["mean", "count", "std"])
w["ci95"] = 1.96 * w["std"] / np.sqrt(w["count"])
OUT["weather_ci"] = w["ci95"].round(1).to_dict()

OUT["traffic_x_area"] = df.groupby(["Traffic", "Area"], observed=True)["Delivery_Time"].mean().unstack().round(0).to_dict("index")
OUT["traffic_x_vehicle"] = df.groupby(["Traffic", "Vehicle"], observed=True)["Delivery_Time"].mean().unstack().round(0).to_dict("index")
OUT["motorcycle_gap_by_traffic"] = {str(t): r(df[(df.Traffic == t) & (df.Vehicle == "Motorcycle")]["Delivery_Time"].mean()
                                              - df[(df.Traffic == t) & (df.Vehicle == "Scooter")]["Delivery_Time"].mean())
                                    for t in TRAFFIC}
OUT["semi_urban"] = {"orders": int((df.Area == "Semi-Urban").sum()),
                     "jam_share_pct": r((df[df.Area == "Semi-Urban"]["Traffic"] == "Jam").mean() * 100),
                     "avg_time": r(df[df.Area == "Semi-Urban"]["Delivery_Time"].mean()),
                     "on_time_pct": r(df[df.Area == "Semi-Urban"]["on_time"].mean() * 100)}
OUT["traffic_by_hour"] = df.groupby("hour")["Traffic"].agg(lambda s: s.mode()[0]).to_dict()
OUT["traffic_hour_purity_pct"] = r(df.groupby("hour")["Traffic"].agg(lambda s: s.value_counts(normalize=True).max()).mean() * 100)
OUT["weekend"] = {"weekend": r(df[df.is_weekend == 1]["Delivery_Time"].mean()), "weekday": r(df[df.is_weekend == 0]["Delivery_Time"].mean()),
                  "p": float(stats.ttest_ind(df[df.is_weekend == 1]["Delivery_Time"], df[df.is_weekend == 0]["Delivery_Time"], equal_var=False).pvalue)}

rb = pd.cut(non["Agent_Rating"], [0, 4.0, 4.5, 4.7, 4.9, 5.0], labels=["<=4.0", "4.1-4.5", "4.6-4.7", "4.8-4.9", "5.0"])
OUT["rating_bands"] = non.groupby(rb, observed=True)["Delivery_Time"].agg(["count", "mean"]).round(1).to_dict("index")
ab = pd.cut(non["Agent_Age"], [19, 24, 29, 34, 39], labels=["20-24", "25-29", "30-34", "35-39"])
OUT["age_bands"] = non.groupby(ab, observed=True)["Delivery_Time"].agg(["count", "mean"]).round(1).to_dict("index")
OUT["age_step"] = {"age29": r(non[non.Agent_Age == 29]["Delivery_Time"].mean()), "age30": r(non[non.Agent_Age == 30]["Delivery_Time"].mean())}
OUT["rating_low_share_pct"] = r((df["Agent_Rating"] <= 4.0).mean() * 100)

df["store_area"] = df["Store_Latitude"].round(1).astype(str) + "," + df["Store_Longitude"].round(1).astype(str)
sa = df.groupby("store_area").agg(orders=("Order_ID", "count"), avg_time=("Delivery_Time", "mean"),
                                  lat=("Store_Latitude", "mean"), lon=("Store_Longitude", "mean")).round(2)
sa.to_csv(BASE / "data" / "store_area_summary.csv")
OUT["store_areas"] = {"count": len(sa), "orders_min": int(sa.orders.min()), "orders_median": int(sa.orders.median()),
                      "orders_max": int(sa.orders.max()), "time_min": r(sa.avg_time.min()), "time_max": r(sa.avg_time.max()),
                      "time_std": r(sa.avg_time.std()), "time_std_orders_ge_100": r(sa[sa.orders >= 100].avg_time.std()),
                      "areas_ge_100": int((sa.orders >= 100).sum()),
                      "top5_share_pct": r(sa.orders.nlargest(5).sum() / n * 100)}

corr_cols = ["Delivery_Time", "distance_km", "Agent_Age", "Agent_Rating", "prep_minutes", "hour", "Traffic_code", "est_cost"]
cm = df[corr_cols].corr()
cm_s = df[corr_cols].corr(method="spearman")
OUT["corr_pearson"] = cm["Delivery_Time"].round(2).to_dict()
OUT["corr_spearman"] = cm_s["Delivery_Time"].round(2).to_dict()
OUT["corr_non_grocery"] = non[["Delivery_Time", "distance_km", "Agent_Age", "Agent_Rating", "prep_minutes"]].corr()["Delivery_Time"].round(2).to_dict()
slope, intercept, rv, pv, se = stats.linregress(non["distance_km"], non["Delivery_Time"])
OUT["distance_regression"] = {"slope": r(slope, 2), "intercept": r(intercept), "r2": round(rv ** 2, 3)}

daily = df.groupby("Order_Date").agg(orders=("Order_ID", "count"), avg_time=("Delivery_Time", "mean"),
                                     on_time=("on_time", "mean"), peak_share=("is_peak_hour", "mean"),
                                     jam_share=("Traffic", lambda s: (s == "Jam").mean()))
full_index = pd.date_range(daily.index.min(), daily.index.max())
daily = daily.reindex(full_index)
daily.index.name = "date"
daily["day_type"] = np.where(daily["peak_share"] > 0.75, "Evening-only", "All-day")
daily.loc[daily["orders"].isna(), "day_type"] = np.nan
daily.to_csv(BASE / "data" / "daily_summary.csv")
obs = daily.dropna()
df["day_type"] = df["Order_Date"].map(daily["day_type"])
OUT["daily"] = {"mean": r(obs.orders.mean()), "min": int(obs.orders.min()), "max": int(obs.orders.max()),
                "missing_days": int(daily.orders.isna().sum()),
                "avg_time_min": r(obs.avg_time.min()), "avg_time_max": r(obs.avg_time.max())}

dtp = obs.groupby("day_type").agg(days=("orders", "count"), orders=("orders", "mean"), avg_time=("avg_time", "mean"),
                                  on_time=("on_time", "mean"), peak=("peak_share", "mean"), jam=("jam_share", "mean"))
OUT["day_types"] = {k: {"days": int(v.days), "orders": r(v.orders, 0), "avg_time": r(v.avg_time), "on_time": r(v.on_time * 100),
                        "peak_pct": r(v.peak * 100), "jam_pct": r(v.jam * 100)} for k, v in dtp.iterrows()}
pairs = [(obs.index[i], obs.index[i + 1]) for i in range(len(obs) - 1) if (obs.index[i + 1] - obs.index[i]).days == 1]
OUT["alternation"] = {"pairs": len(pairs), "alternating": int(sum(obs.loc[x, "day_type"] != obs.loc[y, "day_type"] for x, y in pairs))}
for per, mask in [("before_gap", obs.index <= "2022-02-18"), ("after_gap", obs.index >= "2022-03-01")]:
    OUT[f"orders_{per}"] = {k: r(v, 0) for k, v in obs[mask].groupby("day_type")["orders"].mean().items()}
OUT["orders_per_day_overall"] = {"before_gap": r(obs[obs.index <= "2022-02-18"].orders.mean(), 0), "after_gap": r(obs[obs.index >= "2022-03-01"].orders.mean(), 0)}
OUT["corr_daily_orders_vs_time"] = r(obs.orders.corr(obs.avg_time), 2)
wd = obs.groupby(obs.index.dayofweek).orders.mean().round(0)
OUT["weekday_orders"] = {int(k): int(v) for k, v in wd.items()}
OUT["phase_shift"] = {"21_mar": daily.loc["2022-03-21", "day_type"], "23_mar": daily.loc["2022-03-23", "day_type"]}

peak_n = df[df.is_peak_hour == 1].groupby(["Order_Date", "day_type"]).size().reset_index(name="n")
OUT["evening_orders_per_hour"] = {k: r(v / 7, 0) for k, v in peak_n.groupby("day_type")["n"].mean().items()}
OUT["peak_time_by_day_type"] = {k: r(v) for k, v in df[df.is_peak_hour == 1].groupby("day_type")["Delivery_Time"].mean().items()}
OUT["morning_share_by_day_type"] = {k: r(v * 100) for k, v in df.groupby("day_type")["hour"].apply(lambda x: x.between(8, 11).mean()).items()}
hp = df.groupby(["day_type", "hour"]).size().unstack(0).fillna(0)

hourly = df.groupby("hour").agg(orders=("Order_ID", "count"), avg_time=("Delivery_Time", "mean"), on_time=("on_time", "mean"))
hourly["on_time"] *= 100
hourly.round(1).to_csv(BASE / "data" / "hourly_summary.csv")
OUT["hourly"] = hourly.round(1).to_dict("index")
OUT["peak_vs_off"] = {"peak_time": r(df[df.is_peak_hour == 1]["Delivery_Time"].mean()), "off_time": r(df[df.is_peak_hour == 0]["Delivery_Time"].mean()),
                      "peak_orders_share": r(df["is_peak_hour"].mean() * 100), "peak_on_time": r(df[df.is_peak_hour == 1]["on_time"].mean() * 100),
                      "off_on_time": r(df[df.is_peak_hour == 0]["on_time"].mean() * 100)}

seg = daily.loc["2022-03-01":"2022-03-21", "orders"].astype(float)
ac = acf(seg, nlags=10, fft=False)
OUT["acf"] = {int(i): r(v, 2) for i, v in enumerate(ac) if i > 0}
dec = seasonal_decompose(seg, model="additive", period=2)
res_ = dec.resid.dropna()
OUT["decomposition"] = {"segment_days": len(seg), "seasonal_amplitude": r(abs(dec.seasonal.max()), 0),
                        "trend_start": r(dec.trend.dropna().iloc[0], 0), "trend_end": r(dec.trend.dropna().iloc[-1], 0),
                        "resid_std": r(res_.std(), 1), "series_std": r(seg.std(), 1),
                        "seasonal_explained_pct": r((1 - res_.var() / seg.loc[res_.index].var()) * 100)}
OUT["decomposition"]["trend_slope_per_day"] = r(np.polyfit(np.arange(len(dec.trend.dropna())), dec.trend.dropna().values, 1)[0], 2)

pk = df[df.is_peak_hour == 1]
OUT["peak_by_day_type_traffic"] = {t: {k: r(v) for k, v in pk[pk.Traffic == t].groupby("day_type")["Delivery_Time"].mean().items()} for t in ["Low", "Medium", "Jam"]}
OUT["peak_distance_by_day_type"] = {k: r(v) for k, v in pk.groupby("day_type")["distance_km"].mean().items()}
OUT["distance_by_part_of_day"] = {k: r(v) for k, v in df.groupby("part_of_day", observed=True)["distance_km"].mean().items()}
OUT["peak_hour_range_by_day_type"] = {k: [r(v.min()), r(v.max())] for k, v in pk.groupby("day_type")["Delivery_Time"]}

ols = smf.ols('Delivery_Time ~ C(Traffic, Treatment("Low")) + C(Weather, Treatment("Sunny")) + C(Vehicle, Treatment("Motorcycle")) '
              '+ C(Area, Treatment("Metropolitan")) + distance_km + Agent_Age + Agent_Rating + is_grocery + C(day_type, Treatment("All-day"))', data=df).fit()
ci = ols.conf_int()
names = {}
for term in ols.params.index:
    if term == "Intercept":
        continue
    lab = term.replace('C(Traffic, Treatment("Low"))[T.', "Traffic: ").replace('C(Weather, Treatment("Sunny"))[T.', "Weather: ") \
              .replace('C(Vehicle, Treatment("Motorcycle"))[T.', "Vehicle: ").replace('C(Area, Treatment("Metropolitan"))[T.', "Area: ") \
              .replace('C(day_type, Treatment("All-day"))[T.', "Day type: ").replace("]", "")
    names[lab] = {"coef": r(ols.params[term], 2), "lo": r(ci.loc[term, 0], 2), "hi": r(ci.loc[term, 1], 2), "p": float(ols.pvalues[term])}
OUT["ols"] = names
OUT["ols_r2"] = round(float(ols.rsquared), 3)
OUT["ols_adj_r2"] = round(float(ols.rsquared_adj), 3)
OUT["ols_rmse"] = r(np.sqrt(ols.mse_resid), 1)

late_o = df[df.on_time == 0]
OUT["late_orders"] = int(len(late_o))
OUT["late_share"] = {"Traffic": (late_o["Traffic"].value_counts(normalize=True) * 100).round(1).reindex(TRAFFIC).to_dict(),
                     "Area": (late_o["Area"].value_counts(normalize=True) * 100).round(1).to_dict(),
                     "Vehicle": (late_o["Vehicle"].value_counts(normalize=True) * 100).round(1).to_dict()}
OUT["order_share"] = {"Traffic": (df["Traffic"].value_counts(normalize=True) * 100).round(1).reindex(TRAFFIC).to_dict(),
                      "Vehicle": (df["Vehicle"].value_counts(normalize=True) * 100).round(1).to_dict()}
OUT["jam_peak_late_share"] = r(((late_o["Traffic"] == "Jam") | (late_o["Traffic"] == "High")).mean() * 100)

OUT["cost_by_traffic"] = df.groupby("Traffic", observed=True)["est_cost"].mean().round(1).to_dict()
OUT["cost_by_traffic"] = {str(k): v for k, v in OUT["cost_by_traffic"].items()}
OUT["cost_by_vehicle"] = df.groupby("Vehicle")["est_cost"].mean().round(1).to_dict()
OUT["cost_by_category_grocery"] = {"grocery": r(gro["est_cost"].mean()), "other": r(non["est_cost"].mean())}
OUT["cost_components"] = {"rider": r(df.cost_rider.mean()), "vehicle": r(df.cost_vehicle.mean())}
OUT["cost_per_km_all"] = r(df["est_cost"].sum() / df["distance_km"].sum())

mc, sc = df[df.Vehicle == "Motorcycle"], df[df.Vehicle == "Scooter"]
sc_gap = mc["Delivery_Time"].mean() - sc["Delivery_Time"].mean()
jam = df[df.Traffic == "Jam"]
high = df[df.Traffic == "High"]
jam_gap = jam["Delivery_Time"].mean() - high["Delivery_Time"].mean()
su = df[df.Area == "Semi-Urban"]
mt = df[df.Area == "Metropolitan"]
su_gap = su["Delivery_Time"].mean() - mt["Delivery_Time"].mean()
lowr = df[df.Agent_Rating <= 4.0]
lr_gap = non[non.Agent_Rating <= 4.0]["Delivery_Time"].mean() - non[non.Agent_Rating > 4.5]["Delivery_Time"].mean()


def scen(gap, orders):
    return {"gap_min": r(gap), "orders": int(orders), "minutes_saved": int(gap * orders),
            "rider_hours": int(gap * orders / 60), "rs_saved": int(gap * orders * RIDER_RS_PER_MIN)}


OUT["scenarios"] = {
    "motorcycle_like_scooter": scen(sc_gap, len(mc)),
    "jam_like_high": scen(jam_gap, len(jam)),
    "semi_urban_like_metro": scen(su_gap, len(su)),
    "low_rated_like_good": scen(lr_gap, (non.Agent_Rating <= 4.0).sum()),
}
OUT["total_rider_minutes"] = int(df["Delivery_Time"].sum())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
bins = np.arange(0, 281, 5)
axes[0].hist(non["Delivery_Time"], bins=bins, alpha=0.8, color="#4F81BD", label="Other categories")
axes[0].hist(gro["Delivery_Time"], bins=bins, alpha=0.9, color="#C0504D", label="Grocery")
axes[0].axvline(SLA, color="black", ls=":", label=f"Target {SLA} min")
axes[0].set_title("Delivery time distribution")
axes[0].set_xlabel("Minutes")
axes[0].set_ylabel("Orders")
axes[0].legend(fontsize=8)
cm_ = df.groupby("Category")["Delivery_Time"].mean().sort_values()
axes[1].barh(cm_.index, cm_.values, color=["#C0504D" if c == "Grocery" else "#4F81BD" for c in cm_.index])
axes[1].set_title("Average delivery time by category")
axes[1].set_xlabel("Minutes")
axes[1].tick_params(axis="y", labelsize=8)
save(fig, "01_distribution_and_category.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
pal = dict(zip(TRAFFIC, sns.color_palette("YlOrRd", 4)))
sns.boxplot(x="Traffic", y="Delivery_Time", hue="Traffic", data=df, order=TRAFFIC, hue_order=TRAFFIC, palette=pal, legend=False, ax=axes[0])
axes[0].axhline(SLA, color="black", ls=":")
axes[0].set_title("Delivery time by traffic level")
axes[0].set_ylabel("Minutes")
ot = df.groupby("Traffic", observed=True)["on_time"].mean().mul(100).reindex(TRAFFIC)
axes[1].bar(ot.index, ot.values, color=[pal[t] for t in ot.index])
for i, v in enumerate(ot.values):
    axes[1].text(i, v + 1, f"{v:.0f}%", ha="center")
axes[1].set_title(f"On-time rate (within {SLA} min)")
axes[1].set_ylim(0, 105)
axes[1].set_ylabel("%")
save(fig, "02_traffic.png")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
ws = df.groupby("Weather")["Delivery_Time"].mean().sort_values()
axes[0].barh(ws.index, ws.values, xerr=[OUT["weather_ci"][k] for k in ws.index], color="#4F81BD", capsize=3)
axes[0].set_title("Weather (mean, 95% CI)")
axes[0].set_xlabel("Minutes")
vs = df.groupby("Vehicle")["Delivery_Time"].mean().sort_values()
axes[1].bar(vs.index, vs.values, color="#9BBB59")
axes[1].set_title("Vehicle")
axes[1].set_ylabel("Minutes")
for i, v in enumerate(vs.values):
    axes[1].text(i, v + 2, f"{v:.0f}", ha="center")
ars = df.groupby("Area")["Delivery_Time"].mean().sort_values()
axes[2].bar(ars.index, ars.values, color="#F79646")
axes[2].set_title("Area")
axes[2].tick_params(axis="x", rotation=20)
for i, v in enumerate(ars.values):
    axes[2].text(i, v + 3, f"{v:.0f}", ha="center")
save(fig, "03_weather_vehicle_area.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
sns.heatmap(df.groupby(["Traffic", "Area"], observed=True)["Delivery_Time"].mean().unstack().reindex(TRAFFIC), annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[0], cbar=False)
axes[0].set_title("Avg minutes: traffic × area")
axes[0].set_ylabel("")
sns.heatmap(df.groupby(["Traffic", "Vehicle"], observed=True)["Delivery_Time"].mean().unstack().reindex(TRAFFIC), annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1], cbar=False)
axes[1].set_title("Avg minutes: traffic × vehicle")
axes[1].set_ylabel("")
save(fig, "04_interaction_heatmaps.png")

fig, ax = plt.subplots(figsize=(6.3, 4.9))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (Pearson)")
save(fig, "05_correlation_heatmap.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
axes[0].hexbin(non["distance_km"], non["Delivery_Time"], gridsize=30, cmap="Blues", mincnt=1)
xs = np.linspace(non["distance_km"].min(), non["distance_km"].max(), 50)
axes[0].plot(xs, slope * xs + intercept, color="red", lw=2, label=f"Trend: +{slope:.1f} min per km (R² {rv**2:.2f})")
axes[0].set_title("Distance vs delivery time (non-grocery)")
axes[0].set_xlabel("Distance (km)")
axes[0].set_ylabel("Minutes")
axes[0].legend(fontsize=8)
rbm = non.groupby(rb, observed=True)["Delivery_Time"].mean()
axes[1].bar(rbm.index.astype(str), rbm.values, color="#8064A2")
for i, v in enumerate(rbm.values):
    axes[1].text(i, v + 2, f"{v:.0f}", ha="center")
axes[1].set_title("By agent rating (non-grocery)")
axes[1].set_xlabel("Agent rating")
axes[1].set_ylabel("Minutes")
save(fig, "06_distance_and_rating.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ag = non.groupby("Agent_Age")["Delivery_Time"].mean()
axes[0].plot(ag.index, ag.values, marker="o", color="#1F3864")
axes[0].axvline(29.5, color="red", ls="--", alpha=0.6)
axes[0].set_title("Avg delivery time by agent age (non-grocery)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Minutes")
abm = non.groupby(ab, observed=True)["Delivery_Time"].mean()
axes[1].bar(abm.index.astype(str), abm.values, color="#F79646")
for i, v in enumerate(abm.values):
    axes[1].text(i, v + 2, f"{v:.0f}", ha="center")
axes[1].set_title("By age band")
axes[1].set_xlabel("Age band")
save(fig, "07_agent_age.png")

fig, ax = plt.subplots(figsize=(9.5, 3.6))
ax.plot(daily.index, daily["orders"], color="grey", lw=0.8, zorder=1)
for tname, col in [("All-day", "#4F81BD"), ("Evening-only", "#C0504D")]:
    m_ = daily["day_type"] == tname
    ax.scatter(daily.index[m_], daily["orders"][m_], color=col, s=22, label=f"{tname} day", zorder=2)
ax.axvspan(pd.Timestamp("2022-02-19"), pd.Timestamp("2022-02-28"), color="grey", alpha=0.2, label="No data (10 days)")
ax.axvline(pd.Timestamp("2022-03-22"), color="grey", alpha=0.5, ls="--")
ax.set_title("Shipment volume: orders per day (two alternating day types)")
ax.set_ylabel("Orders")
ax.legend(fontsize=8, ncol=3, loc="lower right")
fig.autofmt_xdate()
save(fig, "08_daily_orders.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(range(1, 11), ac[1:], color=["#C0504D" if i % 2 else "#4F81BD" for i in range(1, 11)])
axes[0].axhline(0, color="black", lw=0.8)
axes[0].axhline(1.96 / np.sqrt(len(seg)), color="grey", ls="--")
axes[0].axhline(-1.96 / np.sqrt(len(seg)), color="grey", ls="--")
axes[0].set_title("Autocorrelation of daily orders (1-21 March)")
axes[0].set_xlabel("Lag (days)")
w_ = 0.4
xh = hp.index.values
axes[1].bar(xh - w_ / 2, hp["All-day"].values, w_, label="All-day days", color="#4F81BD")
axes[1].bar(xh + w_ / 2, hp["Evening-only"].values, w_, label="Evening-only days", color="#C0504D")
axes[1].set_title("Orders by hour on each type of day")
axes[1].set_xlabel("Hour")
axes[1].legend(fontsize=8)
save(fig, "09_autocorrelation_and_day_types.png")

fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.2), sharex=True)
for ax, s_, t, c in zip(axes, [dec.observed, dec.trend, dec.seasonal, dec.resid], ["Observed", "Trend", "Seasonal (2-day)", "Residual"],
                        ["#1F3864", "#C0504D", "#9BBB59", "#8064A2"]):
    ax.plot(s_.index, s_.values, color=c, marker="o", ms=2.5, lw=1)
    ax.set_ylabel(t, fontsize=8)
axes[0].set_title("Decomposition of daily orders (1-21 March 2022)")
fig.autofmt_xdate()
save(fig, "10_decomposition.png")

fig, ax = plt.subplots(figsize=(9, 3.8))
colors = [pal[OUT["traffic_by_hour"][h]] for h in hourly.index]
ax.bar(hourly.index, hourly["orders"], color=colors)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Orders")
ax.set_title("Orders per hour (bar colour = traffic level) and average delivery time (line)")
ax2 = ax.twinx()
ax2.plot(hourly.index, hourly["avg_time"], color="black", marker="o", lw=2)
ax2.set_ylabel("Avg minutes")
ax2.grid(False)
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=pal[t], label=t) for t in TRAFFIC], title="Traffic", fontsize=8, loc="upper left")
save(fig, "11_hourly_volume_and_time.png")

fig, ax = plt.subplots(figsize=(6.2, 5.2))
sc_ = ax.scatter(sa["lon"], sa["lat"], s=sa["orders"] / 6, c=sa["avg_time"], cmap="YlOrRd", alpha=0.8, edgecolor="grey", vmin=105, vmax=140)
plt.colorbar(sc_, ax=ax, label="Avg delivery time (min)")
ax.set_title("Store areas: size = orders, colour = avg delivery time")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
save(fig, "12_store_areas.png")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
ct = df.groupby("Traffic", observed=True)[["cost_rider", "cost_vehicle"]].mean().reindex(TRAFFIC)
ct.plot(kind="bar", stacked=True, ax=axes[0], color=["#4F81BD", "#F79646"], rot=0)
axes[0].set_title("Estimated cost per delivery by traffic (Rs)")
axes[0].set_xlabel("")
axes[0].legend(["Rider time", "Vehicle running"], fontsize=8)
cv = df.groupby("Vehicle")[["cost_rider", "cost_vehicle"]].mean()
cv.plot(kind="bar", stacked=True, ax=axes[1], color=["#4F81BD", "#F79646"], rot=0, legend=False)
axes[1].set_title("By vehicle (Rs)")
axes[1].set_xlabel("")
save(fig, "13_estimated_cost.png")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
cmp_ = pd.DataFrame({"Share of all orders": pd.Series(OUT["order_share"]["Traffic"]), "Share of late orders": pd.Series(OUT["late_share"]["Traffic"])})
cmp_.plot(kind="bar", ax=axes[0], color=["#B7C7E0", "#C0504D"], rot=0)
axes[0].set_title(f"Traffic: orders vs late orders (>{SLA} min)")
axes[0].set_ylabel("%")
axes[0].legend(fontsize=8)
cmp2 = pd.DataFrame({"Share of all orders": pd.Series(OUT["order_share"]["Vehicle"]), "Share of late orders": pd.Series(OUT["late_share"]["Vehicle"])})
cmp2.plot(kind="bar", ax=axes[1], color=["#B7C7E0", "#C0504D"], rot=0, legend=False)
axes[1].set_title("Vehicle: orders vs late orders")
axes[1].set_ylabel("%")
save(fig, "14_late_order_sources.png")

fig, ax = plt.subplots(figsize=(7.5, 5.2))
co = pd.DataFrame(OUT["ols"]).T.sort_values("coef")
ax.errorbar(co["coef"], range(len(co)), xerr=[co["coef"] - co["lo"], co["hi"] - co["coef"]], fmt="o", color="#1F3864", capsize=3)
ax.axvline(0, color="red", lw=1)
ax.set_yticks(range(len(co)))
ax.set_yticklabels(co.index, fontsize=8)
ax.set_xlabel("Extra minutes vs reference category (or per unit), with 95% CI")
ax.set_title(f"Multivariable regression (R² = {OUT['ols_r2']})")
save(fig, "15_regression_coefficients.png")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


with open(BASE / "eda_summary.json", "w") as f:
    json.dump(clean(OUT), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Saved eda_summary.json and", len(list(FIG.glob("*.png"))), "figures")
