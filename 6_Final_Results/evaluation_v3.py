"""Version-3 evaluation: metrics, volatility regimes, MCS (full sample and PM sample) and Diebold-Mariano tests
(base vs PM specification), with Holm and Benjamini-Hochberg adjustments.

Same conventions as the evaluation scripts of the first version (regimi_vol.py, mcs.py, mcs_pm.py, diebold_mariano.py),
which it replaces:
- all models are evaluated on the dates common to every forecast file of a given horizon and sample;
- direction and volatility regime are defined on the VIX at the forecast origin t (h trading days before the target);
- MCS: R statistic, 5000 block-bootstrap replications, block length max(h, ceil(sqrt(T))), alpha = 5%;
- DM: two-sided, Bartlett long-run variance, Harvey et al. (1997) correction.
Outputs in 6_Final_Results/Results/v3.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from dieboldmariano import dm_test
from model_confidence_set import ModelConfidenceSet
from numba import njit


@njit
def _seed_numba(seed):
    np.random.seed(seed) # the MCS bootstrap draws from numba's generator, which np.random.seed does not reach


_seed_numba(42)

ROOT = Path(__file__).resolve().parent.parent
FC = ROOT / "5_Forecasts_&_Error_Metrics"
OUT = ROOT / "6_Final_Results" / "Results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 5, 22)
REGIMES = ["Low (<20)", "Medium (20-30)", "High (>30)"]

vix = pd.read_csv(ROOT / "0_Dataset" / "Data" / "Clean" / "dataset.csv", parse_dates=["Date"]).set_index("Date")["Close_VIX"]


def full_files(h):
    eco, ml, dl = (FC / "Normale" / d / f"h{h}" for d in ("Modelli_econometrici", "Machine_Learning", "Deep_Learning"))
    # The ARIMA(1,1,1) is no longer part of the comparison (version 3)
    files = {"RW": eco / f"randomwalk_h{h}.csv", "HAR": eco / f"har_h{h}.csv", "HAR-X": eco / "v3" / f"harx_h{h}.csv"}
    for name, folder, stem in (("RF", ml, "random_forest"), ("XGBoost", ml, "xgboost"),
                               ("LSTM", dl, "lstm"), ("TFT", dl, "tft")):
        files[f"{name}-GS-CV"] = folder / "v3" / f"{stem}_gridsearch_h{h}.csv"
        files[f"{name}-BO"] = folder / "v3" / f"{stem}_bayesoptimization_h{h}.csv"
    return files


def pm_pairs(h):
    eco, ml, dl = (FC / "With_Polymarket" / d / f"h{h}" for d in ("Modelli_econometrici", "Machine_Learning", "Deep_Learning"))
    pairs = {"HAR": (eco, "har_h{h}_{e}.csv"), "HAR-X": (eco / "v3", "harx_h{h}_{e}.csv")}
    for name, folder, stem, bo in (("RF", ml, "random_forest", "bayesopt"), ("XGBoost", ml, "xgboost", "bayesopt"),
                                   ("LSTM", dl, "lstm", "bayesoptimization"), ("TFT", dl, "tft", "bayesoptimization")):
        pairs[f"{name}-GS-CV"] = (folder / "v3", stem + "_gridsearch_h{h}_{e}.csv")
        pairs[f"{name}-BO"] = (folder / "v3", stem + "_" + bo + "_h{h}_{e}.csv")
    return {m: tuple(folder / pat.format(h=h, e=e) for e in ("senza_PM", "con_PM")) for m, (folder, pat) in pairs.items()}


def load(path):
    d = pd.read_csv(path).dropna()
    # ISO dates (yyyy-mm-dd) in most files, dd/mm/yyyy in some older files. format="mixed" with dayfirst=True
    # would read 2022-10-06 as 10 June 2022, so each format is parsed explicitly.
    iso = d["Date"].str.match(r"^\d{4}-\d{2}-\d{2}")
    d["Date"] = pd.to_datetime(d["Date"].where(iso), format="%Y-%m-%d").fillna(
        pd.to_datetime(d["Date"].where(~iso), format="%d/%m/%Y"))
    d = d.sort_values("Date").set_index("Date")
    assert not d.index.duplicated().any(), path
    return d[["Actual", "Forecast"]]


def align(frames, h):
    """Common dates, consistency of the actual values, VIX at the origin and volatility regime."""
    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    actual = None
    fc = {}
    for m, d in frames.items():
        d = d.loc[common]
        if actual is None:
            actual = d["Actual"]
        elif np.abs(d["Actual"].values - actual.values).max() > 1e-5:
            raise ValueError(f"Actual values differ for {m}")
        fc[m] = d["Forecast"]
    df = pd.DataFrame(fc)
    dates = vix.index.union(actual.index)
    pos = dates.get_indexer(df.index) - h
    assert (pos >= 0).all()
    origin = vix.reindex(dates).ffill().iloc[pos]
    df.insert(0, "Origin_VIX", origin.values)
    df.insert(0, "Actual", actual.values)
    df["Regime"] = np.select([df.Origin_VIX < 20, df.Origin_VIX <= 30], REGIMES[:2], REGIMES[2])
    return df


def metrics(a, f, v0):
    e = f - a
    return {"N": len(a), "MSE": np.mean(e ** 2), "MAE": np.mean(np.abs(e)), "MAPE": 100 * np.mean(np.abs(e / a)),
            "R2": 1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2),
            "QLIKE": np.mean(a / f - np.log(a / f) - 1),
            "DA": 100 * np.mean((f > v0) == (a > v0))}


def mcs(losses, h):
    T = len(losses)
    m = ModelConfidenceSet(losses.reset_index(drop=True), n_boot=5000, alpha=0.05,
                           block_len=max(h, int(np.ceil(np.sqrt(T)))), method="R")
    m.compute()
    r = m.results()
    return r["pvalues"] if "pvalues" in r else r.iloc[:, 0]


def qlike_loss(a, f):
    return a / f - np.log(a / f) - 1


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    running = 1.0
    for k, i in enumerate(order):
        running = min(running, p[i] * n / (n - k))
        adj[i] = running
    return adj


tab_full, tab_reg, tab_mcs, tab_pm_loss, tab_dm, tab_mcs_pm, tab_reg_pm = [], [], [], [], [], [], []

for h in HORIZONS:
    # ------------------------------------------------------------------ full sample
    df = align({m: load(p) for m, p in full_files(h).items()}, h)
    models = [c for c in df.columns if c not in ("Actual", "Origin_VIX", "Regime")]
    for m in models:
        tab_full.append({"h": h, "Model": m, **metrics(df.Actual, df[m], df.Origin_VIX)})
        for reg in REGIMES:
            s = df[df.Regime == reg]
            if len(s):
                tab_reg.append({"h": h, "Model": m, "Regime": reg, **metrics(s.Actual, s[m], s.Origin_VIX)})
    p_mse = mcs(pd.DataFrame({m: (df[m] - df.Actual) ** 2 for m in models}), h)
    p_ql = mcs(pd.DataFrame({m: qlike_loss(df.Actual, df[m]) for m in models}), h)
    for m in models:
        tab_mcs.append({"h": h, "Model": m, "p_MSE": p_mse[m], "p_QLIKE": p_ql[m]})
    print(f"h={h}: full sample, {len(df)} common dates, {df.index[0].date()} to {df.index[-1].date()}")

    # ------------------------------------------------------------------ Polymarket sample
    pairs = pm_pairs(h)
    frames = {}
    for m, (f0, f1) in pairs.items():
        frames[m] = load(f0)
        frames[m + " (PM)"] = load(f1)
    dp = align(frames, h) # joint sample, used for the MCS (as in mcs_pm.py)
    for m in pairs:
        # pairwise sample for the DM test and the regimes (as in diebold_mariano.py)
        dq = align({m: frames[m], m + " (PM)": frames[m + " (PM)"]}, h)
        a, f0, f1 = dq.Actual.values, dq[m].values, dq[m + " (PM)"].values
        row = {"h": h, "Model": m, "N": len(a)}
        for lname, lf in (("MSE", lambda x: (a - x) ** 2), ("QLIKE", lambda x: qlike_loss(a, x))):
            l0, l1 = lf(f0), lf(f1)
            row[f"{lname}_base"], row[f"{lname}_PM"] = l0.mean(), l1.mean()
            row[f"d{lname}_%"] = 100 * (l1.mean() - l0.mean()) / l0.mean()
            try:
                stat, p = dm_test(np.zeros_like(a), l1, l0, h=h, variance_estimator="bartlett",
                                  loss=lambda actual, forecast: forecast)
            except Exception:
                stat, p = np.nan, np.nan
            row[f"DM_{lname}"], row[f"p_{lname}"] = stat, p
        tab_dm.append(row)
        tab_pm_loss.append({"h": h, "Model": m, **{k: v for k, v in row.items() if k.endswith(("_base", "_PM"))}})
    specs = list(frames)
    # regimes on the joint sample, so that all specifications are compared on the same origins (as in regimi_vol_pm.py)
    for m in specs:
        for reg in REGIMES:
            s = dp[dp.Regime == reg]
            if len(s):
                tab_reg_pm.append({"h": h, "Model": m, "Regime": reg, **metrics(s.Actual, s[m], s.Origin_VIX)})
    p_mse = mcs(pd.DataFrame({m: (dp[m] - dp.Actual) ** 2 for m in specs}), h)
    p_ql = mcs(pd.DataFrame({m: qlike_loss(dp.Actual, dp[m]) for m in specs}), h)
    for m in specs:
        tab_mcs_pm.append({"h": h, "Model": m, "p_MSE": p_mse[m], "p_QLIKE": p_ql[m]})
    print(f"h={h}: PM sample, {len(dp)} common dates, {dp.index[0].date()} to {dp.index[-1].date()}")

# multiple-testing adjustment over all DM tests (models x horizons x losses)
dm = pd.DataFrame(tab_dm)
pv = np.concatenate([dm.p_MSE.values, dm.p_QLIKE.values])
ok = ~np.isnan(pv)
for name, fn in (("Holm", holm), ("BH", bh)):
    adj = np.full_like(pv, np.nan)
    adj[ok] = fn(pv[ok])
    dm[f"p_MSE_{name}"], dm[f"p_QLIKE_{name}"] = adj[:len(dm)], adj[len(dm):]
print(f"DM tests adjusted jointly: {ok.sum()}")

for name, t in (("metrics_full", pd.DataFrame(tab_full)), ("regimes_full", pd.DataFrame(tab_reg)),
                ("mcs_full", pd.DataFrame(tab_mcs)), ("dm_pm", dm), ("losses_pm", pd.DataFrame(tab_pm_loss)),
                ("mcs_pm", pd.DataFrame(tab_mcs_pm)), ("regimes_pm", pd.DataFrame(tab_reg_pm))):
    t.to_csv(OUT / f"{name}.csv", index=False)

pd.set_option("display.width", 250)
full = pd.DataFrame(tab_full)
print("\nMSE / QLIKE / DA, full sample")
print(full.pivot(index="Model", columns="h", values=["MSE", "QLIKE", "DA"]).round(3).loc[list(full_files(1))].to_string())
print("\nMCS p-values, full sample")
print(pd.DataFrame(tab_mcs).pivot(index="Model", columns="h", values=["p_MSE", "p_QLIKE"]).round(4).loc[list(full_files(1))].to_string())
print("\nDM base vs PM")
print(dm[["h", "Model", "N", "dMSE_%", "p_MSE", "p_MSE_Holm", "p_MSE_BH", "dQLIKE_%", "p_QLIKE", "p_QLIKE_Holm", "p_QLIKE_BH"]].round(4).to_string(index=False))
print("\nMCS p-values, PM sample")
print(pd.DataFrame(tab_mcs_pm).pivot(index="Model", columns="h", values=["p_MSE", "p_QLIKE"]).round(4).to_string())
