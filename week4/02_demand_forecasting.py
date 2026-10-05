import sys, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)
res = {}


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


daily = pd.read_csv(DAILY, parse_dates=["date"])
obs = daily.dropna(subset=["orders"]).reset_index(drop=True)
obs["e"] = (obs["day_type"] == "Evening-only").astype(int)
res["alternation_in_observation_order"] = {"pairs": len(obs) - 1, "alternating": int((obs["e"].diff().dropna().abs() == 1).sum())}
S = obs[obs["date"] >= "2022-03-01"].reset_index(drop=True)          # training window: 1 March - 6 April
y, e, dates = S["orders"].values.astype(float), S["e"].values, S["date"]
N = len(S)
res["series"] = {"observed_days": N, "first": str(dates.iloc[0].date()), "last": str(dates.iloc[-1].date()),
                 "missing_inside": 1, "mean": round(float(y.mean()), 1), "min": int(y.min()), "max": int(y.max()),
                 "all_day_mean": round(float(y[e == 0].mean()), 1), "evening_mean": round(float(y[e == 1].mean()), 1),
                 "all_day_std": round(float(y[e == 0].std()), 1), "evening_std": round(float(y[e == 1].std()), 1)}


def f_mean(tr, etr, efut, h):               return np.full(h, tr.mean())
def f_naive(tr, etr, efut, h):              return np.full(h, tr[-1])
def f_snaive7(tr, etr, efut, h):            return np.array([tr[len(tr) - 7 + (i % 7)] for i in range(h)])      # typical weekly assumption
def f_ma7(tr, etr, efut, h):                return np.full(h, tr[-7:].mean())                                   # 7-day moving average
def f_snaive2(tr, etr, efut, h):            return np.array([tr[len(tr) - 2 + (i % 2)] for i in range(h)])


def f_type_mean3(tr, etr, efut, h):
    out = []
    for t_ in efut:
        same = tr[etr == t_][-3:]
        out.append(same.mean())
    return np.array(out)


def f_type_mean_all(tr, etr, efut, h):
    return np.array([tr[etr == t_].mean() for t_ in efut])


def f_hw(tr, etr, efut, h):
    m = ExponentialSmoothing(tr, trend=None, seasonal="add", seasonal_periods=2, initialization_method="estimated").fit()
    return m.forecast(h)


def f_sarimax(tr, etr, efut, h):
    m = SARIMAX(tr, exog=etr.reshape(-1, 1), order=(1, 0, 0), trend="c").fit(disp=False)
    return m.forecast(h, exog=np.asarray(efut).reshape(-1, 1))


MODELS = {"Mean": f_mean, "Naive (yesterday)": f_naive, "Weekly seasonal naive (lag 7)": f_snaive7, "7-day moving average": f_ma7,
          "Seasonal naive (lag 2)": f_snaive2, "Same-type average (last 3)": f_type_mean3, "Same-type average (all)": f_type_mean_all,
          "Holt-Winters (period 2)": f_hw, "Regression + AR(1) errors (day type)": f_sarimax}


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return {"MAE": round(float(np.mean(np.abs(a - p))), 1), "RMSE": round(float(np.sqrt(np.mean((a - p) ** 2))), 1),
            "MAPE": round(float(np.mean(np.abs(a - p) / a) * 100), 1)}


H1 = 14
rows1, preds1 = {}, {k: [] for k in MODELS}
actual1 = y[N - H1:]
for t in range(N - H1, N):
    for name, fn in MODELS.items():
        try:
            p = fn(y[:t], e[:t], e[t:t + 1], 1)[0]
        except Exception:
            p = np.nan
        preds1[name].append(p)
for name in MODELS:
    rows1[name] = metrics(actual1, preds1[name])
res["onestep"] = rows1
res["onestep_days"] = H1

H2 = 8
tr_end = N - H2
rows2, preds2 = {}, {}
actual2 = y[tr_end:]
for name, fn in MODELS.items():
    p = fn(y[:tr_end], e[:tr_end], e[tr_end:], H2)
    preds2[name] = p; rows2[name] = metrics(actual2, p)
res["multistep"] = rows2
res["multistep_days"] = H2
cmp = pd.DataFrame({"1-step MAE": {k: v["MAE"] for k, v in rows1.items()}, "1-step RMSE": {k: v["RMSE"] for k, v in rows1.items()},
                    "1-step MAPE %": {k: v["MAPE"] for k, v in rows1.items()}, "8-day MAE": {k: v["MAE"] for k, v in rows2.items()},
                    "8-day RMSE": {k: v["RMSE"] for k, v in rows2.items()}, "8-day MAPE %": {k: v["MAPE"] for k, v in rows2.items()}})
cmp.to_csv(BASE / "data" / "forecast_model_comparison.csv")
print(cmp.to_string())
best = cmp["8-day RMSE"].add(cmp["1-step RMSE"]).idxmin()
res["best_model"] = best                         # lowest combined RMSE; the top models are within about 3 orders of each other
res["final_model"] = "Same-type average (last 3 days)"
top = ["Same-type average (last 3)", "Same-type average (all)", "Holt-Winters (period 2)",
       "Regression + AR(1) errors (day type)", "Seasonal naive (lag 2)"]
avg_rmse = cmp.loc[top, ["1-step RMSE", "8-day RMSE"]].mean(axis=1)
res["top_models_rmse_spread"] = round(float(avg_rmse.max() - avg_rmse.min()), 1)

H3 = 7
last_type = int(e[-1])
e_future = np.array([(last_type + 1 + i) % 2 for i in range(H3)])
fut_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=H3)
m = SARIMAX(y, exog=e.reshape(-1, 1), order=(1, 0, 0), trend="c").fit(disp=False)
res["sarimax"] = {"const": round(float(m.params[0]), 1), "day_type_effect": round(float(m.params[1]), 1), "ar1": round(float(m.params[2]), 2),
                  "day_type_effect_p": round(float(m.pvalues[1]), 2)}
mean_ = f_type_mean3(y, e, e_future, H3)
errs = np.abs(np.concatenate([actual1 - np.array(preds1["Same-type average (last 3)"]), actual2 - preds2["Same-type average (last 3)"]]))
q80 = float(np.quantile(errs, 0.8))
res["interval_halfwidth_80"] = round(q80, 1)
ci = np.column_stack([mean_ - q80, mean_ + q80])
from statsmodels.stats.diagnostic import acorr_ljungbox
res_in = np.array([y[t_] - y[:t_][e[:t_] == e[t_]][-3:].mean() for t_ in range(6, N)])
res["in_sample_rmse"] = round(float(np.sqrt(np.mean(res_in ** 2))), 1)
res["ljung_box_p"] = round(float(acorr_ljungbox(res_in, lags=[5], return_df=True)["lb_pvalue"].iloc[0]), 3)
res["residual_autocorr_lag1"] = round(float(np.corrcoef(res_in[:-1], res_in[1:])[0, 1]), 2)
fdf = pd.DataFrame({"date": fut_dates, "day_type": np.where(e_future == 1, "Evening-only", "All-day"), "forecast_orders": np.round(mean_, 0),
                    "lower_80": np.round(ci[:, 0], 0), "upper_80": np.round(ci[:, 1], 0)})
fdf.to_csv(BASE / "data" / "forecast_next7.csv", index=False)
res["forecast"] = fdf.assign(date=fdf["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
res["forecast_week_total"] = int(fdf["forecast_orders"].sum())
res["forecast_types"] = {"all_day_days": int((e_future == 0).sum()), "evening_days": int((e_future == 1).sum())}

df = load()
dfs = df[df["Order_Date"] >= "2022-03-01"]
prof = dfs.groupby(["day_type", "hour"]).size().unstack(0).fillna(0)
share = prof / prof.sum()
share.to_csv(BASE / "data" / "hourly_profiles.csv")
res["hourly_share"] = {t: {int(h): round(float(v), 4) for h, v in share[t].items()} for t in share.columns}

fig, ax = plt.subplots(figsize=(9.5, 3.8))
idx = np.arange(N - H1, N)
ax.plot(idx, actual1, color="black", lw=2.5, marker="o", label="Actual")
styles = {"Weekly seasonal naive (lag 7)": ("#C0504D", "--"), "Naive (yesterday)": ("#F79646", "--"), "Seasonal naive (lag 2)": ("#4F81BD", "-"),
          "Holt-Winters (period 2)": ("#8064A2", "-"), "Regression + AR(1) errors (day type)": ("#9BBB59", "-")}
for name, (c_, ls) in styles.items():
    ax.plot(idx, preds1[name], color=c_, ls=ls, marker=".", label=name)
ax.set_xlabel("Observed day (index)"); ax.set_ylabel("Orders"); ax.set_title(f"One-step-ahead forecasts, last {H1} days")
ax.legend(fontsize=7, ncol=2, loc="lower left")
save(fig, "06_backtest_one_step.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
order_ = cmp.sort_values("1-step RMSE").index
axes[0].barh(order_, cmp.loc[order_, "1-step RMSE"], color="#4F81BD"); axes[0].set_title("1-step-ahead RMSE (orders)")
axes[0].invert_yaxis()
axes[1].barh(order_, cmp.loc[order_, "8-day RMSE"], color="#F79646"); axes[1].set_title("8-day-ahead RMSE (orders)")
for a in axes:
    for pt in a.patches: a.text(pt.get_width() + 2, pt.get_y() + pt.get_height() / 2, f"{pt.get_width():.0f}", va="center", fontsize=8)
axes[0].tick_params(axis="y", labelsize=8)
save(fig, "07_forecast_model_comparison.png")

fig, ax = plt.subplots(figsize=(9.5, 3.9))
hist_n = 24
ax.plot(dates.iloc[-hist_n:], y[-hist_n:], color="grey", marker="o", ms=3, label="Observed")
for t_, c_ in [(0, "#4F81BD"), (1, "#C0504D")]:
    mk = e[-hist_n:] == t_
    ax.scatter(dates.iloc[-hist_n:][mk], y[-hist_n:][mk], color=c_, s=24, zorder=3, label=("All-day" if t_ == 0 else "Evening-only") + " day")
ax.plot(fdf["date"], fdf["forecast_orders"], color="#1F3864", marker="s", label="Forecast")
ax.fill_between(fdf["date"], fdf["lower_80"], fdf["upper_80"], color="#1F3864", alpha=0.2, label="80% interval")
ax.set_title("Forecast of daily orders, 7 to 13 April 2022"); ax.set_ylabel("Orders"); ax.legend(fontsize=8, ncol=3, loc="lower left")
fig.autofmt_xdate()
save(fig, "08_forecast_next_week.png")

dump(res, "results_forecast.json")
print("best:", best)
print(fdf.to_string())
