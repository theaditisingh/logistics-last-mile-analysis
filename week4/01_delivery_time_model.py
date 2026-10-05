import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", font_scale=0.95)
res = {}


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


df = load()
train, test = split_random(df)
tr_t, te_t = split_time(df)
res["n"] = {"rows": len(df), "train": len(train), "test": len(test), "temporal_train": len(tr_t), "temporal_test": len(te_t),
            "temporal_test_days": int(te_t["Order_Date"].nunique()), "cutoff": CUTOFF}
y_tr, y_te = train[TARGET], test[TARGET]
cv = KFold(5, shuffle=True, random_state=SEED)

base_pred = np.full(len(y_te), y_tr.mean())
res["baseline"] = {"MAE": round(mean_absolute_error(y_te, base_pred), 2), "RMSE": round(rmse(y_te, base_pred), 2),
                   "R2": round(r2_score(y_te, base_pred), 3)}
rows = []
for name, m in candidates(FEATURES_A).items():
    log(f"CV {name}")
    cvr = cross_validate(m, train[FEATURES_A], y_tr, cv=cv, scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
    m.fit(train[FEATURES_A], y_tr); p = m.predict(test[FEATURES_A])
    m2 = candidates(FEATURES_A)[name]; m2.fit(tr_t[FEATURES_A], tr_t[TARGET]); pt = m2.predict(te_t[FEATURES_A])
    rows.append({"Model": name, "CV_R2": cvr["test_r2"].mean(), "CV_R2_std": cvr["test_r2"].std(),
                 "CV_RMSE": -cvr["test_neg_root_mean_squared_error"].mean(), "CV_MAE": -cvr["test_neg_mean_absolute_error"].mean(),
                 "Test_MAE": mean_absolute_error(y_te, p), "Test_RMSE": rmse(y_te, p), "Test_R2": r2_score(y_te, p),
                 "Time_RMSE": rmse(te_t[TARGET], pt), "Time_R2": r2_score(te_t[TARGET], pt)})
comp = pd.DataFrame(rows).round(3)
res["comparison"] = comp.to_dict("records")
print(comp.to_string(index=False), flush=True)

log("tuning gradient boosting")
t0 = time.time()
gb_grid = GridSearchCV(pipe(HistGradientBoostingRegressor(random_state=SEED), FEATURES_A),
    {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63], "model__min_samples_leaf": [20, 50],
     "model__l2_regularization": [0.0, 1.0]}, cv=3, scoring="neg_root_mean_squared_error", n_jobs=1).fit(train[FEATURES_A], y_tr)
log("tuning random forest")
rf_grid = GridSearchCV(pipe(RandomForestRegressor(n_estimators=60, max_features=0.6, random_state=SEED, n_jobs=1), FEATURES_A),
    {"model__max_depth": [12, 18], "model__min_samples_leaf": [5, 15]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=1).fit(train[FEATURES_A], y_tr)
res["tuning"] = {"seconds": round(time.time() - t0),
                 "gb_best": {k.replace("model__", ""): v for k, v in gb_grid.best_params_.items()}, "gb_cv_rmse_3fold": round(-gb_grid.best_score_, 3),
                 "gb_combinations": len(gb_grid.cv_results_["params"]),
                 "rf_best": {k.replace("model__", ""): v for k, v in rf_grid.best_params_.items()}, "rf_cv_rmse_3fold": round(-rf_grid.best_score_, 3),
                 "rf_combinations": len(rf_grid.cv_results_["params"])}
pd.DataFrame(gb_grid.cv_results_)[["params", "mean_test_score"]].to_csv(BASE / "data" / "tuning_results_gb.csv", index=False)

tuned = {"Tuned Gradient Boosting": gb_grid.best_estimator_, "Tuned Random Forest": rf_grid.best_estimator_}
cand_cv = {r["Model"]: r["CV_RMSE"] for r in rows}
trows = []
for name, m in tuned.items():
    log(f"5-fold CV {name}")
    cvr = cross_validate(m, train[FEATURES_A], y_tr, cv=cv, scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
    cand_cv[name] = -cvr["test_neg_root_mean_squared_error"].mean()
    m.fit(train[FEATURES_A], y_tr); p = m.predict(test[FEATURES_A])
    from sklearn.base import clone
    m2 = clone(m); m2.fit(tr_t[FEATURES_A], tr_t[TARGET]); pt = m2.predict(te_t[FEATURES_A])
    trows.append({"Model": name, "CV_R2": cvr["test_r2"].mean(), "CV_R2_std": cvr["test_r2"].std(),
                  "CV_RMSE": cand_cv[name], "CV_MAE": -cvr["test_neg_mean_absolute_error"].mean(),
                  "Test_MAE": mean_absolute_error(y_te, p), "Test_RMSE": rmse(y_te, p), "Test_R2": r2_score(y_te, p),
                  "Time_RMSE": rmse(te_t[TARGET], pt), "Time_R2": r2_score(te_t[TARGET], pt)})
comp = pd.concat([comp, pd.DataFrame(trows).round(3)], ignore_index=True)
comp.to_csv(BASE / "data" / "model_comparison.csv", index=False)
res["comparison"] = comp.to_dict("records")
best_name = min(cand_cv, key=cand_cv.get)
res["best_model"] = best_name
log(f"best by CV RMSE: {best_name}")
all_models = {**candidates(FEATURES_A), **tuned}
from sklearn.base import clone

final_A = clone(all_models[best_name])
final_A.fit(train[FEATURES_A], y_tr)
base_est = clone(all_models[best_name]).named_steps["model"]
final_B = Pipeline([("prep", preprocessor(FEATURES_B)), ("model", clone(base_est))])
cvB = cross_validate(final_B, train[FEATURES_B], y_tr, cv=cv, scoring=("r2", "neg_root_mean_squared_error"))
final_B.fit(train[FEATURES_B], y_tr)
pA, pB = final_A.predict(test[FEATURES_A]), final_B.predict(test[FEATURES_B])
fB_t = clone(final_B).fit(tr_t[FEATURES_B], tr_t[TARGET]); fA_t = clone(final_A).fit(tr_t[FEATURES_A], tr_t[TARGET])
res["final_A"] = {"MAE": round(mean_absolute_error(y_te, pA), 2), "RMSE": round(rmse(y_te, pA), 2), "R2": round(r2_score(y_te, pA), 3),
                  "temporal_RMSE": round(rmse(te_t[TARGET], fA_t.predict(te_t[FEATURES_A])), 2),
                  "temporal_R2": round(r2_score(te_t[TARGET], fA_t.predict(te_t[FEATURES_A])), 3),
                  "temporal_MAE": round(mean_absolute_error(te_t[TARGET], fA_t.predict(te_t[FEATURES_A])), 2)}
res["final_B"] = {"MAE": round(mean_absolute_error(y_te, pB), 2), "RMSE": round(rmse(y_te, pB), 2), "R2": round(r2_score(y_te, pB), 3),
                  "cv_rmse": round(-cvB["test_neg_root_mean_squared_error"].mean(), 3), "cv_r2": round(cvB["test_r2"].mean(), 3),
                  "temporal_RMSE": round(rmse(te_t[TARGET], fB_t.predict(te_t[FEATURES_B])), 2),
                  "temporal_R2": round(r2_score(te_t[TARGET], fB_t.predict(te_t[FEATURES_B])), 3),
                  "temporal_MAE": round(mean_absolute_error(te_t[TARGET], fB_t.predict(te_t[FEATURES_B])), 2)}
import statsmodels.formula.api as smf
form = ('Delivery_Time ~ C(Traffic, Treatment("Low")) + C(Weather, Treatment("Sunny")) + C(Vehicle, Treatment("Motorcycle")) '
        '+ C(Area, Treatment("Metropolitan")) + distance_km + Agent_Age + Agent_Rating + is_grocery + day_type_evening')
ols = smf.ols(form, data=train).fit()
po = ols.predict(test)
res["ols_week3"] = {"MAE": round(mean_absolute_error(y_te, po), 2), "RMSE": round(rmse(y_te, po), 2), "R2": round(r2_score(y_te, po), 3)}

use_B = res["final_B"]["RMSE"] < res["final_A"]["RMSE"] - 0.5
final, FEATS, pred = (final_B, FEATURES_B, pB) if use_B else (final_A, FEATURES_A, pA)
res["final_uses_day_type"] = bool(use_B)
res["final"] = res["final_B"] if use_B else res["final_A"]
resid = y_te.values - pred
res["residual"] = {"mean": round(float(resid.mean()), 2), "std": round(float(resid.std()), 2),
                   "within_10": round(float((np.abs(resid) <= 10).mean() * 100), 1),
                   "within_15": round(float((np.abs(resid) <= 15).mean() * 100), 1),
                   "within_30": round(float((np.abs(resid) <= 30).mean() * 100), 1)}
t = test.copy(); t["pred"] = pred; t["abs_err"] = np.abs(resid); t["resid"] = resid
res["mae_by_traffic"] = t.groupby("Traffic")["abs_err"].mean().round(1).reindex(TRAFFIC).to_dict()
res["mae_by_grocery"] = {"grocery": round(t[t.is_grocery == 1]["abs_err"].mean(), 1), "other": round(t[t.is_grocery == 0]["abs_err"].mean(), 1)}
res["mae_by_area"] = t.groupby("Area")["abs_err"].mean().round(1).to_dict()
res["bias_by_traffic"] = t.groupby("Traffic")["resid"].mean().round(1).reindex(TRAFFIC).to_dict()
res["features_used"] = FEATS

log("permutation importance")
pi = permutation_importance(final, test[FEATS], y_te, n_repeats=5, random_state=SEED, scoring="r2")
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
res["importance"] = imp.round(4).to_dict()
imp.round(4).to_csv(BASE / "data" / "feature_importance.csv")

log("cross-validated predictions for thresholds and buffers")
cv_pred = cross_val_predict(clone(final), train[FEATS], y_tr, cv=cv)
cv_res = y_tr.values - cv_pred
late_tr = (y_tr.values > SLA)
best_f1, best_thr = 0, 0
for thr in range(100, 181, 2):
    fl = cv_pred >= thr
    tp = (fl & late_tr).sum(); fp = (fl & ~late_tr).sum(); fn = (~fl & late_tr).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
    if f1 > best_f1: best_f1, best_thr = f1, thr
late_te = (y_te.values > SLA)
fl = pred >= best_thr
tp, fp, fn, tn = int((fl & late_te).sum()), int((fl & ~late_te).sum()), int((~fl & late_te).sum()), int((~fl & ~late_te).sum())
res["early_warning"] = {"threshold": best_thr, "late_orders": int(late_te.sum()), "flagged": int(fl.sum()), "caught": tp, "false_alarms": fp,
                        "missed": fn, "recall": round(tp / (tp + fn) * 100, 1), "precision": round(tp / (tp + fp) * 100, 1),
                        "auc": round(float(roc_auc_score(late_te, pred)), 3), "base_rate": round(float(late_te.mean() * 100), 1)}
sweep = []
for thr in range(100, 181, 5):
    fl_ = pred >= thr
    tp_ = (fl_ & late_te).sum(); fp_ = (fl_ & ~late_te).sum(); fn_ = (~fl_ & late_te).sum()
    sweep.append((thr, tp_ / max(tp_ + fp_, 1) * 100, tp_ / max(tp_ + fn_, 1) * 100, fl_.mean() * 100))

q_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
curve = []
for q in q_grid:
    buf = float(np.quantile(cv_res, q))
    prom = pred + buf
    curve.append({"q": q, "buffer": round(buf, 1), "on_time": round(float((y_te.values <= prom).mean() * 100), 1), "avg_promise": round(float(prom.mean()), 1)})
res["promise_curve"] = curve
buf90 = float(np.quantile(cv_res, 0.90)); prom = pred + buf90
res["promise"] = {"buffer": round(buf90, 1), "dynamic_on_time": round(float((y_te.values <= prom).mean() * 100), 1),
                  "avg_promise": round(float(prom.mean()), 1), "fixed_on_time": round(float((y_te.values <= SLA).mean() * 100), 1),
                  "min_promise": round(float(prom.min()), 1), "max_promise": round(float(prom.max()), 1)}
g = pd.DataFrame({"traffic": t["Traffic"].values, "grocery": t["is_grocery"].values, "ok_dyn": y_te.values <= prom,
                  "ok_fix": y_te.values <= SLA, "promise": prom})
res["promise_by_traffic"] = {k: {"fixed": round(v.ok_fix.mean() * 100, 1), "dynamic": round(v.ok_dyn.mean() * 100, 1), "promise": round(v.promise.mean(), 1)}
                             for k, v in g.groupby("traffic")}
res["promise_by_grocery"] = {("grocery" if k == 1 else "other"): {"fixed": round(v.ok_fix.mean() * 100, 1), "dynamic": round(v.ok_dyn.mean() * 100, 1),
                                                                 "promise": round(v.promise.mean(), 1)} for k, v in g.groupby("grocery")}
seg_buf = pd.Series(cv_res).groupby(train["Traffic"].values).quantile(0.9)
prom_seg = pred + t["Traffic"].map(seg_buf).values
res["promise_segment"] = {"on_time": round(float((y_te.values <= prom_seg).mean() * 100), 1), "avg_promise": round(float(prom_seg.mean()), 1),
                          "buffers": seg_buf.round(1).to_dict()}
np.save(BASE / "data" / "cv_residuals.npy", cv_res)

order = comp.sort_values("CV_RMSE")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
colors = ["#1F3864" if m == best_name else "#4F81BD" for m in order["Model"]]
axes[0].barh(order["Model"], order["CV_RMSE"], color=colors)
axes[0].axvline(res["baseline"]["RMSE"], color="red", ls="--", label=f"Mean baseline ({res['baseline']['RMSE']})")
axes[0].set_title("5-fold cross-validated RMSE (min)"); axes[0].legend(fontsize=8)
for i, v in enumerate(order["CV_RMSE"]): axes[0].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
w = 0.38; yy = np.arange(len(order))
axes[1].barh(yy - w / 2, order["Test_RMSE"], w, label="Random test set", color="#4F81BD")
axes[1].barh(yy + w / 2, order["Time_RMSE"], w, label="Later period (temporal)", color="#F79646")
axes[1].set_yticks(yy); axes[1].set_yticklabels([]); axes[1].set_title("Test RMSE (min)"); axes[1].legend(fontsize=8)
save(fig, "01_model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
hb = axes[0].hexbin(y_te, pred, gridsize=40, cmap="Blues", mincnt=1)
lim = [0, 280]; axes[0].plot(lim, lim, "r--", label="Perfect prediction"); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel("Actual delivery time (min)"); axes[0].set_ylabel("Predicted (min)"); axes[0].set_title(f"Predicted vs actual (R² {res['final']['R2']})"); axes[0].legend(fontsize=8)
sns.histplot(resid, bins=50, kde=True, ax=axes[1], color="#8064A2"); axes[1].axvline(0, color="black")
axes[1].set_title("Residuals (actual − predicted)"); axes[1].set_xlabel("Minutes")
save(fig, "02_predicted_vs_actual_residuals.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
top = imp.head(9)[::-1]
axes[0].barh(top.index, top.values, color="#1F3864"); axes[0].set_title("Permutation importance (drop in R²)")
mt = t.groupby("Traffic")["abs_err"].mean().reindex(TRAFFIC)
mg = t.groupby("is_grocery")["abs_err"].mean()
labels = TRAFFIC + ["Grocery", "Other"]
vals = list(mt.values) + [mg[1], mg[0]]
axes[1].bar(labels, vals, color=["#F2C46D"] * 4 + ["#C0504D", "#4F81BD"]); axes[1].set_title("Mean absolute error by group (min)")
axes[1].tick_params(axis="x", labelsize=8)
for i, v in enumerate(vals): axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
save(fig, "03_importance_and_error.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sw = pd.DataFrame(sweep, columns=["thr", "precision", "recall", "flagged"])
axes[0].plot(sw.thr, sw.precision, marker="o", label="Precision"); axes[0].plot(sw.thr, sw.recall, marker="s", label="Recall")
axes[0].axvline(best_thr, color="grey", ls="--"); axes[0].set_xlabel("Flag order if predicted time ≥ (min)"); axes[0].set_ylabel("%")
axes[0].set_title("Late-delivery early warning"); axes[0].legend(fontsize=8)
cdf = pd.DataFrame(curve)
axes[1].plot(cdf.avg_promise, cdf.on_time, marker="o", color="#1F3864", label="Model-based promise")
for _, r_ in cdf.iterrows(): axes[1].annotate(f"{int(r_.q*100)}%", (r_.avg_promise, r_.on_time), textcoords="offset points", xytext=(4, -10), fontsize=8)
axes[1].scatter([SLA], [res["promise"]["fixed_on_time"]], color="red", zorder=5, label=f"Fixed {SLA}-min promise")
axes[1].set_xlabel("Average promised time (min)"); axes[1].set_ylabel("On-time %"); axes[1].set_title("Promise length vs on-time rate"); axes[1].legend(fontsize=8)
save(fig, "04_early_warning_and_promise.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
gt = g.groupby("traffic")[["ok_fix", "ok_dyn"]].mean().mul(100).reindex(TRAFFIC)
gt.columns = [f"Fixed {SLA}-min promise", "Model-based promise"]
gt.plot(kind="bar", ax=axes[0], color=["#C0504D", "#4F81BD"], rot=0); axes[0].set_title("On-time rate by traffic level"); axes[0].set_ylabel("%"); axes[0].set_xlabel("")
axes[0].legend(fontsize=8, loc="lower left"); axes[0].set_ylim(0, 105)
g.groupby("traffic")["promise"].mean().reindex(TRAFFIC).plot(kind="bar", ax=axes[1], color="#8064A2", rot=0)
axes[1].axhline(SLA, color="black", ls=":"); axes[1].set_title("Average model-based promise (min)"); axes[1].set_xlabel("")
save(fig, "05_promise_by_traffic.png")

dump(res, "results_model.json")
log("done part A")
