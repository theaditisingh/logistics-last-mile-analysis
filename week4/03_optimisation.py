import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

sns.set_theme(style="whitegrid", font_scale=0.95)
res = {}
RIDER_RS_PER_HOUR = 90.0                 # same assumption as Week 3 (Rs 1.5 per minute)
RUN_RS_PER_KM = {"Motorcycle": 3.0, "Scooter": 2.5, "Van": 8.0}


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


fc = pd.read_csv(BASE / "data" / "forecast_next7.csv", parse_dates=["date"])
share = pd.read_csv(BASE / "data" / "hourly_profiles.csv", index_col=0)
share.index = share.index.astype(int)
rm = json.load(open(BASE / "results_model.json"))
rf_ = json.load(open(BASE / "results_forecast.json"))

S_MIN, UTIL = 30, 0.8        # ASSUMPTIONS: rider time needed per delivery (min) and target utilisation
HOURS = list(range(8, 25))    # 08:00 ... 24:00 (hour 24 = the 00:00 orders)
vol = {"All-day": fc[fc.day_type == "All-day"]["forecast_orders"].mean(), "Evening-only": fc[fc.day_type == "Evening-only"]["forecast_orders"].mean()}
n_days = {"All-day": int((fc.day_type == "All-day").sum()), "Evening-only": int((fc.day_type == "Evening-only").sum())}


def requirement(day_type, s_min=S_MIN, util=UTIL):
    sh = share[day_type].copy()
    sh.index = [24 if h == 0 else h for h in sh.index]
    orders = pd.Series({h: vol[day_type] * sh.get(h, 0.0) for h in HOURS})
    return orders, np.ceil(orders * s_min / 60 / util).astype(int)


SHIFTS = [(a, L) for L in (4, 6, 8) for a in HOURS if a + L - 1 <= 24]


def solve_schedule(req):
    c = np.array([L + 0.25 for a, L in SHIFTS])
    A = np.zeros((len(HOURS), len(SHIFTS)))
    for j, (a, L) in enumerate(SHIFTS):
        for h in range(a, a + L):
            A[HOURS.index(h), j] = 1
    r = milp(c, constraints=LinearConstraint(A, lb=np.asarray([req[h] for h in HOURS], float), ub=np.inf),
             integrality=np.ones(len(SHIFTS)), bounds=Bounds(0, np.inf))
    x = np.round(r.x).astype(int)
    staffed = pd.Series(A @ x, index=HOURS)
    return x, staffed, int((x * np.array([L for a, L in SHIFTS])).sum()), int(x.sum())


def policies(s_min=S_MIN):
    reqs = {t: requirement(t, s_min)[1] for t in vol}
    req_max = pd.concat(reqs.values(), axis=1).max(axis=1)
    wts = pd.Series(n_days) / sum(n_days.values())
    req_avg = np.ceil(sum(reqs[t] * wts[t] for t in reqs)).astype(int)
    out = {}
    sched = {}
    for t in reqs:
        x, st, rh, ns = solve_schedule(reqs[t]); sched[t] = (x, st); out[f"type_{t}"] = {"rider_hours": rh, "shifts": ns, "peak_riders": int(reqs[t].max())}
    for nm, rq in [("max", req_max), ("avg", req_avg)]:
        x, st, rh, ns = solve_schedule(rq); sched[nm] = (x, st)
        out[nm] = {"rider_hours": rh, "shifts": ns, "peak_riders": int(rq.max())}
    week = {}
    week["one_size_max"] = {"rider_hours": out["max"]["rider_hours"] * 7, "shortfall_rider_hours": 0}
    sf = sum(n_days[t] * float(np.maximum(reqs[t] - sched["avg"][1], 0).sum()) for t in reqs)
    week["one_size_avg"] = {"rider_hours": out["avg"]["rider_hours"] * 7, "shortfall_rider_hours": round(sf, 1)}
    week["by_day_type"] = {"rider_hours": sum(n_days[t] * out[f"type_{t}"]["rider_hours"] for t in reqs), "shortfall_rider_hours": 0}
    return reqs, sched, out, week


reqs, sched, out, week = policies()
res["assumptions"] = {"rider_minutes_per_delivery": S_MIN, "utilisation": UTIL, "rider_rs_per_hour": RIDER_RS_PER_HOUR,
                      "forecast_orders": {k: round(v) for k, v in vol.items()}, "days": n_days}
res["requirements"] = {t: {"orders_by_hour": {int(h): round(float(v), 1) for h, v in requirement(t)[0].items()},
                           "riders_by_hour": {int(h): int(v) for h, v in reqs[t].items()}} for t in reqs}
res["schedules"] = out
res["week"] = week
res["week_savings"] = {
    "vs_one_size_max_hours": week["one_size_max"]["rider_hours"] - week["by_day_type"]["rider_hours"],
    "vs_one_size_max_pct": round((1 - week["by_day_type"]["rider_hours"] / week["one_size_max"]["rider_hours"]) * 100, 1),
    "vs_one_size_max_rs": round((week["one_size_max"]["rider_hours"] - week["by_day_type"]["rider_hours"]) * RIDER_RS_PER_HOUR),
    "avg_shortfall_hours": week["one_size_avg"]["shortfall_rider_hours"],
    "avg_shortfall_pct_of_need": round(week["one_size_avg"]["shortfall_rider_hours"] / sum(n_days[t] * reqs[t].sum() for t in reqs) * 100, 1),
    "avg_policy_extra_hours_vs_day_type": week["one_size_avg"]["rider_hours"] - week["by_day_type"]["rider_hours"],
}
sens = {}
for s_ in (20, 30, 40):
    _, _, o_, w_ = policies(s_)
    sens[s_] = {"one_size_max": w_["one_size_max"]["rider_hours"], "by_day_type": w_["by_day_type"]["rider_hours"],
                "saving_pct": round((1 - w_["by_day_type"]["rider_hours"] / w_["one_size_max"]["rider_hours"]) * 100, 1),
                "peak_all_day": o_["type_All-day"]["peak_riders"], "peak_evening": o_["type_Evening-only"]["peak_riders"]}
res["sensitivity"] = sens
plan = pd.DataFrame({"hour": HOURS})
for t in reqs:
    plan[f"orders_{t}"] = requirement(t)[0].values.round(1); plan[f"required_{t}"] = reqs[t].values; plan[f"scheduled_{t}"] = sched[t][1].values
plan.to_csv(BASE / "data" / "staffing_plan.csv", index=False)
res["shift_plan"] = {t: [{"start": SHIFTS[j][0], "length": SHIFTS[j][1], "riders": int(sched[t][0][j])} for j in range(len(SHIFTS)) if sched[t][0][j] > 0] for t in reqs}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, t, col in zip(axes, ["All-day", "Evening-only"], ["#4F81BD", "#C0504D"]):
    ax.step(HOURS, reqs[t].values, where="post", color="black", lw=2, label="Required riders")
    ax.bar(HOURS, sched[t][1].values, align="edge", width=1.0, color=col, alpha=0.35, label="Scheduled (optimal)")
    ax.set_title(f"{t} day (forecast {round(vol[t])} orders)"); ax.set_xlabel("Hour of day (24 = 00:00)")
    ax.set_xticks(range(8, 25, 2)); ax.set_xlim(8, 25); ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Riders")
save(fig, "09_staffing_plan.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
labs = ["One schedule for\nall days (peak)", "One schedule for\nall days (average)", "Schedule by\nday type"]
vals = [week["one_size_max"]["rider_hours"], week["one_size_avg"]["rider_hours"], week["by_day_type"]["rider_hours"]]
bars = axes[0].bar(labs, vals, color=["#C0504D", "#F79646", "#9BBB59"]); axes[0].set_title("Rider-hours for the forecast week"); axes[0].tick_params(axis="x", labelsize=8)
for b_, v in zip(bars, vals): axes[0].text(b_.get_x() + b_.get_width() / 2, v + 10, f"{v:,}", ha="center", fontsize=9)
shorts = [0, week["one_size_avg"]["shortfall_rider_hours"], 0]
bars = axes[1].bar(labs, shorts, color=["#C0504D", "#F79646", "#9BBB59"]); axes[1].set_title("Unmet rider-hours (staff below requirement)"); axes[1].tick_params(axis="x", labelsize=8)
for b_, v in zip(bars, shorts): axes[1].text(b_.get_x() + b_.get_width() / 2, v + 3, f"{v:,.0f}", ha="center", fontsize=9)
save(fig, "10_staffing_policies.png")

df = load()
use_B = rm["final_uses_day_type"]
FEATS = FEATURES_B if use_B else FEATURES_A
bm = rm["best_model"]
if bm == "Tuned Gradient Boosting":
    est = HistGradientBoostingRegressor(random_state=SEED, **rm["tuning"]["gb_best"])
elif bm == "Tuned Random Forest":
    est = RandomForestRegressor(n_estimators=60, max_features=0.6, random_state=SEED, n_jobs=1, **rm["tuning"]["rf_best"])
else:
    est = candidates(FEATS)[bm].named_steps["model"]
model = Pipeline([("prep", preprocessor(FEATS)), ("model", est)]).fit(df[FEATS], df[TARGET])
VEH = ["Motorcycle", "Scooter", "Van"]
cf = {}
for v in VEH:
    d_ = df[FEATS].copy(); d_["Vehicle"] = v
    cf[v] = model.predict(d_)
cf = pd.DataFrame(cf)
actual_pred = np.array([cf.loc[i, v] for i, v in zip(cf.index, df["Vehicle"].values)])
df["dist_band"] = pd.cut(df["distance_km"], [0, 6, 12, 25], labels=["short (<6 km)", "medium (6-12)", "long (>12)"])
df["segment"] = df["Traffic"].astype(str) + " | " + df["dist_band"].astype(str)
segs = sorted(df["segment"].unique())
n_s = df.groupby("segment").size().reindex(segs).values
T = np.array([[cf.loc[df["segment"] == s, v].mean() for v in VEH] for s in segs])           # mean predicted minutes
N_v = df["Vehicle"].value_counts().reindex(VEH).values
cur_n = np.array([[((df["segment"] == s) & (df["Vehicle"] == v)).sum() for v in VEH] for s in segs])
cur_cost = float((cur_n * T).sum())
ns_, nv_ = len(segs), len(VEH)
c_ = T.flatten()
A_eq, b_eq = [], []
for i in range(ns_):                       # every order in a segment gets one vehicle
    row = np.zeros(ns_ * nv_); row[i * nv_:(i + 1) * nv_] = 1; A_eq.append(row); b_eq.append(n_s[i])
for j in range(nv_):                       # fleet size (orders handled per vehicle type) stays as today
    row = np.zeros(ns_ * nv_); row[j::nv_] = 1; A_eq.append(row); b_eq.append(N_v[j])
lp = linprog(c_, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
x = lp.x.reshape(ns_, nv_)
opt_cost = float(lp.fun)
all_scooter = float((n_s * T[:, 1]).sum())
tot_actual = float(actual_pred.sum())
res["vehicle"] = {"model": bm, "orders": len(df), "predicted_total_minutes_current": round(cur_cost), "predicted_total_minutes_optimal": round(opt_cost),
                  "minutes_saved": round(cur_cost - opt_cost), "pct_saved": round((cur_cost - opt_cost) / cur_cost * 100, 2),
                  "avg_min_saved_per_order": round((cur_cost - opt_cost) / len(df), 2), "rider_hours_saved": round((cur_cost - opt_cost) / 60),
                  "rs_saved": round((cur_cost - opt_cost) * 1.5), "all_scooter_minutes_saved": round(cur_cost - all_scooter),
                  "all_scooter_pct": round((cur_cost - all_scooter) / cur_cost * 100, 2), "segments": ns_,
                  "fleet_orders": {v: int(n) for v, n in zip(VEH, N_v)}}
gap = {}
for tr in TRAFFIC:
    m_ = df["Traffic"] == tr
    gap[tr] = {"motorcycle_minus_scooter": round(float((cf.loc[m_, "Motorcycle"] - cf.loc[m_, "Scooter"]).mean()), 1),
               "motorcycle_minus_van": round(float((cf.loc[m_, "Motorcycle"] - cf.loc[m_, "Van"]).mean()), 1)}
res["vehicle_gap_by_traffic"] = gap
rows = []
for tr in TRAFFIC:
    idx = [i for i, s in enumerate(segs) if s.startswith(tr)]
    n_tot = n_s[idx].sum()
    rows.append({"traffic": tr, "orders": int(n_tot), "moto_current_pct": round(cur_n[idx, 0].sum() / n_tot * 100, 1),
                 "moto_optimal_pct": round(x[idx, 0].sum() / n_tot * 100, 1),
                 "scooter_van_current_pct": round(cur_n[idx][:, 1:].sum() / n_tot * 100, 1), "scooter_van_optimal_pct": round(x[idx][:, 1:].sum() / n_tot * 100, 1)})
res["vehicle_alloc"] = rows

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
al = pd.DataFrame(rows).set_index("traffic")
w_ = 0.38; xx = np.arange(4)
axes[0].bar(xx - w_ / 2, al["moto_current_pct"], w_, label="Current", color="#C0504D")
axes[0].bar(xx + w_ / 2, al["moto_optimal_pct"], w_, label="Optimised", color="#4F81BD")
axes[0].set_xticks(xx); axes[0].set_xticklabels(TRAFFIC); axes[0].set_title("Share of orders given to motorcycles (%)"); axes[0].legend(fontsize=8)
gp = pd.DataFrame(gap).T
axes[1].bar(gp.index, gp["motorcycle_minus_scooter"], color="#F79646"); axes[1].set_title("Model: extra minutes on a motorcycle vs a scooter")
for i, v in enumerate(gp["motorcycle_minus_scooter"]): axes[1].text(i, v + 0.2, f"{v:.1f}", ha="center")
save(fig, "11_vehicle_allocation.png")

def hav(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def dmat(P):
    n = len(P); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = hav(P[i], P[j])
    return D


def rlen(order, D):
    path = [0] + list(order) + [0]
    return sum(D[path[i], path[i + 1]] for i in range(len(path) - 1))


def nn_route(D):
    left, cur, order = set(range(1, len(D))), 0, []
    while left:
        nxt = min(left, key=lambda j: D[cur, j]); order.append(nxt); left.remove(nxt); cur = nxt
    return order


def two_opt(order, D):
    best, bl, imp = order[:], rlen(order, D), True
    while imp:
        imp = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                l = rlen(cand, D)
                if l < bl - 1e-9:
                    best, bl, imp = cand, l, True
    return best, bl


CAP = 6                                           # maximum stops per rider route
df["store_area"] = df["Store_Latitude"].round(1).astype(str) + "," + df["Store_Longitude"].round(1).astype(str)
win = df.groupby(["store_area", "Order_Date", "hour"]).filter(lambda g: len(g) >= 8)
groups = list(win.groupby(["store_area", "Order_Date", "hour"]))
stats_ = {k: [] for k in ["random_listed", "listed", "sweep_listed", "sweep_opt"]}
longest = {k: [] for k in stats_}
nstops, nroutes = [], []
example = None
rng = np.random.default_rng(SEED)
for gi, (key, g) in enumerate(groups):
    P_all = g[["Drop_Latitude", "Drop_Longitude"]].values
    hub = np.array([g["Store_Latitude"].mean(), g["Store_Longitude"].mean()])
    n = len(P_all); k = math.ceil(n / CAP)
    chunks_a = [list(range(i, min(i + CAP, n))) for i in range(0, n, CAP)]
    ang = np.arctan2(P_all[:, 0] - hub[0], P_all[:, 1] - hub[1])
    srt = np.argsort(ang)
    chunks_b = [list(c) for c in np.array_split(srt, k)]
    perm = rng.permutation(n); chunks_r = [list(perm[i:i + CAP]) for i in range(0, n, CAP)]
    res_g = {"random_listed": [], "listed": [], "sweep_listed": [], "sweep_opt": []}
    for nm, chunks, opt in [("random_listed", chunks_r, False), ("listed", chunks_a, False), ("sweep_listed", chunks_b, False), ("sweep_opt", chunks_b, True)]:
        for ch in chunks:
            P = np.vstack([hub, P_all[ch]]); D = dmat(P)
            order = list(range(1, len(P)))
            if opt:
                order, l = two_opt(nn_route(D), D)
            else:
                l = rlen(order, D)
            res_g[nm].append(l)
    for nm in stats_:
        stats_[nm].append(sum(res_g[nm])); longest[nm].append(max(res_g[nm]))
    nstops.append(n); nroutes.append(k)
    if example is None and n >= 14:
        example = (hub, P_all, chunks_a, chunks_b)

tot = {k: float(np.sum(v)) for k, v in stats_.items()}
res["routing"] = {"windows": len(groups), "orders": int(np.sum(nstops)), "routes": int(np.sum(nroutes)), "capacity": CAP,
                  "total_km": {k: round(v, 1) for k, v in tot.items()},
                  "km_per_order": {k: round(v / np.sum(nstops), 2) for k, v in tot.items()},
                  "avg_longest_route_km": {k: round(float(np.mean(v)), 1) for k, v in longest.items()},
                  "saving_vs_listed_pct": {"sweep_listed": round((1 - tot["sweep_listed"] / tot["listed"]) * 100, 1),
                                           "sweep_opt": round((1 - tot["sweep_opt"] / tot["listed"]) * 100, 1)},
                  "saving_vs_random_pct": round((1 - tot["sweep_opt"] / tot["random_listed"]) * 100, 1),
                  "sequencing_only_pct": round((1 - tot["sweep_opt"] / tot["sweep_listed"]) * 100, 1),
                  "km_saved": round(tot["listed"] - tot["sweep_opt"], 1),
                  "rs_saved_at_3_per_km": round((tot["listed"] - tot["sweep_opt"]) * 3.0),
                  "avg_orders_per_window": round(float(np.mean(nstops)), 1), "max_orders_per_window": int(np.max(nstops)),
                  "share_of_all_orders_pct": round(float(np.sum(nstops)) / len(df) * 100, 1)}
per_win = np.array(stats_["listed"]) - np.array(stats_["sweep_opt"])
res["routing"]["windows_improved_pct"] = round(float((per_win > 1e-9).mean() * 100), 1)

hub, P_all, ca, cb = example
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True, sharey=True)
cols = sns.color_palette("tab10", 10)
for ax, chunks, title, opt in [(axes[0], ca, "Before: orders split in listed order", False), (axes[1], cb, "After: sweep assignment + optimised order", True)]:
    tot_len = 0
    for ci, ch in enumerate(chunks):
        P = np.vstack([hub, P_all[ch]]); D = dmat(P)
        if opt:
            order, l = two_opt(nn_route(D), D)
        else:
            order = list(range(1, len(P))); l = rlen(order, D)
        tot_len += l
        path = P[[0] + order + [0]]
        ax.plot(path[:, 1], path[:, 0], "-o", ms=4, color=cols[ci % 10], lw=1.2)
    ax.scatter(hub[1], hub[0], marker="*", s=220, color="black", zorder=5)
    ax.set_title(f"{title}\n(total {tot_len:.1f} km)", fontsize=9); ax.set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
save(fig, "12_routes_before_after.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
labs = ["Random\nassignment", "Listed\norder", "Sweep +\nlisted order", "Sweep + optimised\nroute"]
colors = ["#C0504D", "#F79646", "#4F81BD", "#9BBB59"]
axes[0].bar(labs, [res["routing"]["km_per_order"][k] for k in stats_], color=colors); axes[0].set_title("Rider km per order")
axes[1].bar(labs, [res["routing"]["avg_longest_route_km"][k] for k in stats_], color=colors); axes[1].set_title("Longest route in a window (km)")
for a in axes:
    a.tick_params(axis="x", labelsize=8)
    for p_ in a.patches: a.text(p_.get_x() + p_.get_width() / 2, p_.get_height() + 0.1, f"{p_.get_height():.1f}", ha="center", fontsize=9)
save(fig, "13_routing_results.png")

dump(res, "results_optimisation.json")
print(json.dumps(clean(res["week_savings"]), indent=1)); print(json.dumps(clean(res["vehicle"]), indent=1)); print(json.dumps(clean(res["routing"]), indent=1))
