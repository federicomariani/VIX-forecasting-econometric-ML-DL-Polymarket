"""LaTeX tables and regime heatmaps for the paper, from the outputs of evaluation_v3.py.

Writes Results/v3/tables_v3.tex, with one table environment per paper label, preceded by a line
'%%% LABEL <label>', and the figures vol_regimes_v3.png and vol_regimes_pm_v3.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "6_Final_Results" / "Results" / "v3"
H = (1, 5, 22)
HCOLS = {1: "1 day", 5: "5 days", 22: "22 days"}
MODELS = ["RW", "HAR", "HAR-X", "RF-GS-CV", "RF-BO", "XGBoost-GS-CV", "XGBoost-BO",
          "LSTM-GS-CV", "LSTM-BO", "TFT-GS-CV", "TFT-BO"]
PM_MODELS = MODELS[1:]
REG = ["Low (<20)", "Medium (20-30)", "High (>30)"]
REG_TEX = {"Low (<20)": r"Low ($<20$)", "Medium (20-30)": r"Medium (20--30)", "High (>30)": r"High ($>30$)"}

full = pd.read_csv(RES / "metrics_full.csv")
reg_full = pd.read_csv(RES / "regimes_full.csv")
mcs_full = pd.read_csv(RES / "mcs_full.csv")
dm = pd.read_csv(RES / "dm_pm.csv")
mcs_pm = pd.read_csv(RES / "mcs_pm.csv")
reg_pm = pd.read_csv(RES / "regimes_pm.csv")

out = []


def bold(s, flag):
    return r"\textbf{" + s + "}" if flag else s


def fmt(x, d):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def pct(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}\\%"


def best_flags(values, higher=False):
    v = pd.Series(values, dtype=float)
    if v.notna().sum() == 0:
        return pd.Series(False, index=v.index)
    target = v.max() if higher else v.min()
    return np.isclose(v, target, rtol=0, atol=1e-12) & v.notna()


def header3(cols):
    n = len(cols)
    first = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{HCOLS[h]}}}}}" for h in H)
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(3))
    second = " & ".join(cols * 3)
    return f"& {first} \\\\\n{rules}\n\\textbf{{Model}} & {second} \\\\\n"


# ---------------------------------------------------------------- Table: MSE, MAE, DA
cell = {}
for h in H:
    t = full[full.h == h].set_index("Model").loc[MODELS]
    da = t.DA.where(t.index != "RW")
    for col, d, hi in (("MSE", 4, False), ("MAE", 4, False)):
        f = best_flags(t[col], hi)
        for m in MODELS:
            cell[m, h, col] = bold(fmt(t.at[m, col], d), f[m])
    f = best_flags(da, True)
    for m in MODELS:
        cell[m, h, "DA"] = "N/A" if m == "RW" else bold(pct(da[m]), f[m])
rows = "\n".join(f"{m} & " + " & ".join(cell[m, h, c] for h in H for c in ("MSE", "MAE", "DA")) + r" \\" for m in MODELS)
out.append(("tab:backtest_results", r"""\begin{table}
\scriptsize
\setlength{\tabcolsep}{1.2pt}
\caption{Out-of-sample forecasting performance across forecast horizons ($h \in \{1,5,22\}$ trading days), measured by mean squared error (MSE), mean absolute error (MAE) and directional accuracy (DA). DA is not reported for the RW, since its forecast implies no change in the VIX. Bold values indicate the best model for each horizon and metric.}\label{tab:backtest_results}
\begin{tabular*}{\tblwidth}{@{} L R R R R R R R R R @{}}
\toprule
""" + header3([r"\textbf{MSE}", r"\textbf{MAE}", r"\textbf{DA}"]) + "\\midrule\n" + rows + r"""
\bottomrule
\end{tabular*}
\end{table}"""))

# ---------------------------------------------------------------- Table: MAPE, R2, QLIKE
cell = {}
for h in H:
    t = full[full.h == h].set_index("Model").loc[MODELS]
    for col, d, hi, f_ in (("MAPE", 2, False, pct), ("R2", 4, True, None), ("QLIKE", 4, False, None)):
        f = best_flags(t[col].round(4 if col != "MAPE" else 2), hi)
        for m in MODELS:
            s = f_(t.at[m, col]) if f_ else fmt(t.at[m, col], d)
            cell[m, h, col] = bold(s, f[m])
rows = "\n".join(f"{m} & " + " & ".join(cell[m, h, c] for h in H for c in ("MAPE", "R2", "QLIKE")) + r" \\" for m in MODELS)
out.append(("tab:backtest_results_mape_r2_qlike", r"""\begin{table}[H]
\scriptsize
\setlength{\tabcolsep}{1.2pt}
\caption{Model backtest results (MAPE, $R^2$, and QLIKE) across all time horizons (extended results of \autoref{tab:backtest_results}). $R^2$ is the out-of-sample coefficient of determination, $1-\sum_t (V_{t+h}-\widehat{V}_{t+h|t})^2/\sum_t (V_{t+h}-\bar{V})^2$, computed in the same way for all models. Bold values indicate the best model for each horizon and metric.}\label{tab:backtest_results_mape_r2_qlike}
\begin{tabular*}{\tblwidth}{@{} L R R R R R R R R R @{}}
\toprule
""" + header3([r"\textbf{MAPE}", r"\textbf{$R^2$}", r"\textbf{QLIKE}"]) + "\\midrule\n" + rows + r"""
\bottomrule
\end{tabular*}
\end{table}"""))


# ---------------------------------------------------------------- MCS tables
def mcs_rows(tab, models):
    lines = []
    for m in models:
        vals = []
        for h in H:
            r = tab[(tab.h == h) & (tab.Model == m)].iloc[0]
            vals += [bold(f"{r.p_MSE:.4f}", r.p_MSE >= 0.05), bold(f"{r.p_QLIKE:.4f}", r.p_QLIKE >= 0.05)]
        lines.append(f"{m} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines)


MCS_HEAD = r"""\toprule
& \multicolumn{2}{c}{\textbf{1 day}} & \multicolumn{2}{c}{\textbf{5 days}} & \multicolumn{2}{c}{\textbf{22 days}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\textbf{Model} & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ \\
\midrule
"""
out.append(("tab:mcs", r"""\begin{table}[!h]
\scriptsize
\caption{Model Confidence Set (MCS) $p$-values for the MSE and QLIKE loss functions across forecast horizons ($h \in \{1,5,22\}$ trading days). Bold values indicate inclusion in the MCS at the 5\% significance level.}\label{tab:mcs}
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\tblwidth}{@{} L C C C C C C @{}}
""" + MCS_HEAD + mcs_rows(mcs_full, MODELS) + r"""
\bottomrule
\end{tabular*}
\end{table}"""))

pm_specs = [s for m in PM_MODELS for s in (m, m + " (PM)")]
out.append(("tab:mcs_pm", r"""\begin{table}[!h]
\scriptsize
\caption{Model Confidence Set (MCS) $p$-values for base and Polymarket-augmented (PM) specifications across forecast horizons ($h \in \{1,5,22\}$ trading days), using the MSE and QLIKE loss functions. The MCS is computed jointly over the 20 specifications, on the forecast origins common to all of them. Bold values indicate inclusion in the MCS at the 5\% significance level.}\label{tab:mcs_pm}
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\tblwidth}{@{} L C C C C C C @{}}
""" + MCS_HEAD + mcs_rows(mcs_pm, pm_specs) + r"""
\bottomrule
\end{tabular*}
\end{table}"""))

# ---------------------------------------------------------------- DM tables
lines = []
for m in PM_MODELS:
    vals = []
    for h in H:
        r = dm[(dm.h == h) & (dm.Model == m)].iloc[0]
        vals += [f"{r['dMSE_%']:.2f}", f"{r['dQLIKE_%']:.2f}", bold(f"{r.p_MSE:.4f}", r.p_MSE < 0.05),
                 bold(f"{r.p_QLIKE:.4f}", r.p_QLIKE < 0.05)]
    lines.append(f"{m} & " + " & ".join(vals) + r" \\")
out.append(("tab:dm_test", r"""\begin{table}[!h]
\centering
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\caption{Diebold--Mariano (DM) test results comparing base and Polymarket-augmented specifications across forecast horizons ($h \in \{1,5,22\}$ trading days). The table reports the percentage changes (in \%) in MSE and QLIKE defined in \eqref{eq:delta_loss} and the DM $p$-values under the MSE and QLIKE loss functions. Bold $p$-values indicate nominal significance at the 5\% level. The $p$-values are not adjusted for multiple comparisons.}
\label{tab:dm_test}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l cccc cccc cccc @{}}
\toprule
& \multicolumn{4}{c}{\textbf{1 day}} & \multicolumn{4}{c}{\textbf{5 days}} & \multicolumn{4}{c}{\textbf{22 days}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
\textbf{Model} & $\Delta\text{MSE}$ & $\Delta\text{QLIKE}$ & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ & $\Delta\text{MSE}$ & $\Delta\text{QLIKE}$ & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ & $\Delta\text{MSE}$ & $\Delta\text{QLIKE}$ & $p_{\text{MSE}}$ & $p_{\text{QLIKE}}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}"""))

lines = []
for m in PM_MODELS:
    vals = []
    for h in H:
        r = dm[(dm.h == h) & (dm.Model == m)].iloc[0]
        for c in ("p_MSE_Holm", "p_MSE_BH", "p_QLIKE_Holm", "p_QLIKE_BH"):
            vals.append(bold(f"{r[c]:.4f}", r[c] < 0.05))
    lines.append(f"{m}\n& " + " & ".join(vals) + r" \\")
out.append(("tab:dm_adjusted", r"""\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Adjusted Diebold--Mariano test $p$-values using the
Holm and Benjamini--Hochberg (BH) procedures across all 60 tests.
Bold values indicate statistical significance at the 5\% level
after the corresponding correction (extended results of \autoref{tab:dm_test}).}
\label{tab:dm_adjusted}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l cccc cccc cccc @{}}
\toprule
& \multicolumn{4}{c}{\textbf{1 day}}
& \multicolumn{4}{c}{\textbf{5 days}}
& \multicolumn{4}{c}{\textbf{22 days}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
\textbf{Model}
& $p_{\text{MSE}}^{H}$ & $p_{\text{MSE}}^{BH}$
& $p_{\text{QLIKE}}^{H}$ & $p_{\text{QLIKE}}^{BH}$
& $p_{\text{MSE}}^{H}$ & $p_{\text{MSE}}^{BH}$
& $p_{\text{QLIKE}}^{H}$ & $p_{\text{QLIKE}}^{BH}$
& $p_{\text{MSE}}^{H}$ & $p_{\text{MSE}}^{BH}$
& $p_{\text{QLIKE}}^{H}$ & $p_{\text{QLIKE}}^{BH}$ \\
\midrule
""" + "\n\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\begin{flushleft}
\footnotesize
Notes: $H$ denotes the Holm correction and $BH$ denotes the
Benjamini--Hochberg correction. Both procedures are applied jointly
to the 60 unadjusted DM $p$-values. Values below 0.05 are statistically
significant after the corresponding correction.
\end{flushleft}
\end{table}"""))

# ---------------------------------------------------------------- PM average losses
lines = []
for m in PM_MODELS:
    for suffix, tag in ((" (PM)", "PM"), ("", "base")):
        vals = []
        for h in H:
            r = dm[(dm.h == h) & (dm.Model == m)].iloc[0]
            vals += [f"{r['MSE_' + tag]:.4f}", f"{r['QLIKE_' + tag]:.4f}"]
        lines.append(f"{m}{suffix} & " + " & ".join(vals) + r" \\")
out.append(("tab:backtest_results_mse_qlike", r"""\begin{table}[H]
\scriptsize
\caption{Out-of-sample MSE and QLIKE by model and forecast horizon
($h\in\{1,5,22\}$ trading days), for specifications with and without
Polymarket (PM) features. For each model, both specifications are evaluated on the forecast origins
common to the two (extended results of \autoref{tab:dm_test}).}\label{tab:backtest_results_mse_qlike}
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\tblwidth}{@{} L C C C C C C @{}}
\toprule
& \multicolumn{2}{c}{\textbf{1 day}} & \multicolumn{2}{c}{\textbf{5 days}} & \multicolumn{2}{c}{\textbf{22 days}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\textbf{Model} & \textbf{MSE} & \textbf{QLIKE} & \textbf{MSE} & \textbf{QLIKE} & \textbf{MSE} & \textbf{QLIKE} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}"""))


# ---------------------------------------------------------------- regime tables
def regime_table(tab, models, h, label, caption):
    t = tab[tab.h == h]
    flags = {}
    for reg in REG:
        s = t[t.Regime == reg].set_index("Model")
        if s.empty:
            continue
        for col, hi, d in (("MSE", False, 3), ("MAE", False, 3), ("MAPE", False, 2), ("R2", True, 3),
                           ("QLIKE", False, 4), ("DA", True, 2)):
            v = s[col].round(d)
            if col == "DA":
                v = v.where(v.index != "RW")
            if col == "R2" and s.N.iloc[0] < 3:
                v = v * np.nan
            f = best_flags(v, hi)
            for m in s.index:
                flags[m, reg, col] = bool(f[m])
    blocks = []
    for m in models:
        s = t[t.Model == m].set_index("Regime")
        rows = []
        for reg in REG:
            if reg not in s.index:
                continue
            r = s.loc[reg]
            small = r.N < 3
            vals = [str(int(r.N)),
                    bold(fmt(r.MSE, 3), flags[m, reg, "MSE"]), bold(fmt(r.MAE, 3), flags[m, reg, "MAE"]),
                    bold(pct(r.MAPE), flags[m, reg, "MAPE"]),
                    "--" if small else bold(fmt(r.R2, 3), flags[m, reg, "R2"]),
                    bold(fmt(r.QLIKE, 4), flags[m, reg, "QLIKE"]),
                    "N/A" if m == "RW" else bold(pct(r.DA), flags[m, reg, "DA"])]
            rows.append(f"& {REG_TEX[reg]} & " + " & ".join(vals) + r" \\")
        blocks.append(f"\\multirow{{{len(rows)}}}{{*}}{{{m}}}\n" + "\n".join(rows))
    return r"""\begin{table}[H]
\tiny
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\setlength{\tabcolsep}{1.1pt}
\begin{tabular*}{\tblwidth}{@{} L L R R R R R R R @{}}
\toprule
\textbf{Model} & \textbf{Volatility regime} & \textbf{N} & \textbf{MSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{$R^2$} & \textbf{QLIKE} & \textbf{DA} \\
\midrule
""" + "\n\\midrule\n".join(blocks) + r"""
\bottomrule
\end{tabular*}
\end{table}"""


HW = {1: "1-day", 5: "5-day", 22: "22-day"}
for h in H:
    out.append((f"tab:models_h{h}_regimi", regime_table(
        reg_full, MODELS, h, f"tab:models_h{h}_regimi",
        f"Out-of-sample forecasting performance by volatility regime at the {HW[h]} horizon. Bold values indicate "
        r"the best result for each metric within each regime. Directional accuracy is not reported for the RW, "
        r"since its forecast implies no change in the VIX (extended results of \autoref{fig:heatmap_vol_regimes}).")))
for h in H:
    n_high = reg_pm[(reg_pm.h == h) & (reg_pm.Regime == REG[2])].N
    note = (" The high volatility regime contains no forecast origin at this horizon." if n_high.empty else
            f" The high volatility regime contains {int(n_high.iloc[0])} forecast origins, for which $R^2$ is not reported, "
            "and is excluded from the PM comparison in \\autoref{fig:heatmap_vol_regimes_pm}.")
    out.append((f"tab:models_h{h}_regimi_pm", regime_table(
        reg_pm, pm_specs, h, f"tab:models_h{h}_regimi_pm",
        f"Out-of-sample forecasting performance by volatility regime at the {HW[h]} horizon, for models with and "
        "without Polymarket (PM) features, on the forecast origins common to all specifications. Bold values "
        "indicate the best result for each metric within each regime." + note +
        r" Extended results of \autoref{fig:heatmap_vol_regimes_pm}.")))

with open(RES / "tables_v3.tex", "w", encoding="utf-8") as fh:
    for label, tex in out:
        fh.write(f"%%% LABEL {label}\n{tex}\n\n")
print(f"{len(out)} tables written")

# ---------------------------------------------------------------- figures
REG_LAB = {"Low (<20)": "Low\nVIX < 20", "Medium (20-30)": "Medium\n20 ≤ VIX ≤ 30", "High (>30)": "High\nVIX > 30"}

fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), gridspec_kw={"wspace": 0.12})
vmax = reg_full.MSE.max()
for ax, h in zip(axes, H):
    t = reg_full[reg_full.h == h]
    mat = t.pivot(index="Model", columns="Regime", values="MSE").loc[MODELS, REG]
    n = t.groupby("Regime").N.first()
    im = ax.imshow(mat.values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13,
                    color="white" if v > 0.6 * vmax else "black")
    ax.set_xticks(range(3), [f"{REG_LAB[r]}\nN = {n[r]}" for r in REG], fontsize=11)
    ax.set_yticks(range(len(MODELS)), MODELS if h == 1 else [], fontsize=11)
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"h = {h}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Volatility regime", fontsize=12)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.set_label("MSE (common scale)", fontsize=11)
fig.savefig(RES / "vol_regimes_v3.png", dpi=250, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 8.5), gridspec_kw={"wspace": 0.12})
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
for ax, h in zip(axes, H):
    t = reg_pm[reg_pm.h == h]
    base = t[~t.Model.str.endswith("(PM)")].pivot(index="Model", columns="Regime", values="MSE")
    pm = t[t.Model.str.endswith("(PM)")].assign(Model=lambda d: d.Model.str.replace(" (PM)", "", regex=False))
    pm = pm.pivot(index="Model", columns="Regime", values="MSE")
    mat = (100 * (pm - base) / base).loc[PM_MODELS, REG[:2]]
    n = t.groupby("Regime").N.first()
    im = ax.imshow(np.clip(mat.values, -100, 100), cmap="RdYlGn_r", norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=13,
                    color="white" if abs(v) > 70 else "black")
    ax.set_xticks(range(2), [f"{REG_LAB[r]}\nN = {n[r]}" for r in REG[:2]], fontsize=11)
    ax.set_yticks(range(len(PM_MODELS)), PM_MODELS if h == 1 else [], fontsize=11)
    ax.set_xticks(np.arange(-.5, 2), minor=True)
    ax.set_yticks(np.arange(-.5, len(PM_MODELS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"h = {h}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Volatility regime", fontsize=12)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, extend="both")
cb.set_label("Percentage change in MSE (%), color scale clipped at ±100%", fontsize=11)
fig.savefig(RES / "vol_regimes_pm_v3.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("figures written")
