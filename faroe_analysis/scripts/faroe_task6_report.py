#!/usr/bin/env python3
"""Task-6 write-up artefacts: a LaTeX table and a readable figure for the POOLED
("all") held-out scores.

    python faroe_analysis/scripts/faroe_task6_report.py

Reads compare/model_comparison.csv and writes compare/model_comparison_table.tex and
compare/model_comparison_all.png. Reports the three columns asked for: WAIC, predictive
log-score, and smooth TW-CRPS.

FIGURE DESIGN. The default bar chart was unreadable for two reasons, and one of them is
a statistical trap rather than a cosmetic one:

  * the model names overprinted each other on a vertical axis -- fixed by putting the
    models on the y-axis, where long names have room;
  * the CRPS panels were anchored at zero, so a 3% spread across models rendered as five
    identical bars. Simply truncating the axis would exaggerate those same differences
    with a baseline the reader cannot see. Instead each panel plots the DIFFERENCE FROM
    THE BEST MODEL, so zero is a real, meaningful origin, every panel reads "0 = best,
    longer = worse", and the raw value is printed on each bar so nothing is hidden.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator            # noqa: E402


def _tickfmt(v, _pos):
    """Compact tick labels. WAIC runs to six figures, and at default tick density
    those labels collide into an unreadable smear -- the very fault this figure exists
    to fix, so cap the tick count and abbreviate thousands."""
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v/1000:g}k"
    if abs(v) < 0.01:
        return f"{v:g}"
    return f"{v:g}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faroe_setup as F                                            # noqa: E402

# stem -> (display name, family). Families share a colour; the two margin treatments of
# a model are distinguished by fill so the eye groups by model first.
PRETTY = {
    "hw_warm":                 ("HW, sampled margins",     "hw"),
    "hw_fixmargins_stage1":    ("HW, fixed margins",       "hw"),
    "wang_warm":               ("Wang, sampled margins",   "wang"),
    "wang_fixmargins_stage1":  ("Wang, fixed margins",     "wang"),
    "shi_fixmargins_stage1":   ("Shi, fixed margins",      "shi"),
}
FAM_COLOR = {"hw": "#6b6b6b", "wang": "#2e8b57", "shi": "#2b6cb0"}
# (csv key, heading, LaTeX heading, lower_is_better, decimals)
METRICS = [
    ("waic",          "WAIC",                 r"WAIC $\downarrow$",             True,  0),
    ("loglik",        "predictive log-score", r"Log-score $\uparrow$",           False, 1),
    ("twcrps_smooth", "TW-CRPS (smooth)",     r"TW-CRPS$_{\text{smooth}}$ $\downarrow$",
                                                                                 True,  4),
]


def load_all_rows(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["region"] == "all"]
    if not rows:
        raise SystemExit(f"no region='all' rows in {path}")
    out = []
    for r in rows:
        if r["model"] not in PRETTY:
            print(f"note: unknown model stem {r['model']!r} -- using it verbatim")
        name, fam = PRETTY.get(r["model"], (r["model"].replace("_", r" "), "shi"))
        rec = {"stem": r["model"], "name": name, "fam": fam,
               "n_draws": int(float(r["n_draws"]))}
        for key, *_ in METRICS:
            rec[key] = float(r[key]) if r.get(key) not in (None, "") else np.nan
        out.append(rec)
    return out


def _best(rows, key, lower):
    vals = np.array([r[key] for r in rows], float)
    return np.nanmin(vals) if lower else np.nanmax(vals)


def write_tex(rows, path):
    """A standalone booktabs table. Best value per column in bold."""
    lines = [
        r"% Faroe task 6: pooled held-out scores.",
        r"% Requires \usepackage{booktabs} and \usepackage{amsmath} (for \text).",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Held-out predictive performance at the %d Faroe test sites "
        r"(pooled over all three holdout blocks). Arrows give the preferred "
        r"direction; the best value in each column is shown in bold.}"
        % 63,
        r"  \label{tab:faroe-model-comparison}",
        r"  \begin{tabular}{l" + "r" * len(METRICS) + "}",
        r"    \toprule",
        r"    Model & " + " & ".join(m[2] for m in METRICS) + r" \\",
        r"    \midrule",
    ]
    for r in rows:
        cells = []
        for key, _, _, lower, dec in METRICS:
            best = _best(rows, key, lower)
            txt = f"{r[key]:,.{dec}f}"
            cells.append(r"\textbf{" + txt + "}" if np.isclose(r[key], best) else txt)
        lines.append(f"    {r['name']} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def make_figure(rows, path):
    order = sorted(rows, key=lambda r: -r["loglik"])          # best log-score at top
    names = [r["name"] for r in order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 3.5 + 0.32 * len(order)),
                             sharey=True)
    for ax, (key, title, _, lower, dec) in zip(axes, METRICS):
        best = _best(order, key, lower)
        # signed so that 0 = best and larger = worse, in EVERY panel
        delta = [(r[key] - best) if lower else (best - r[key]) for r in order]
        colors = [FAM_COLOR[r["fam"]] for r in order]
        bars = ax.barh(y, delta, color=colors, height=0.68,
                       edgecolor="white", linewidth=0.8)
        # hollow fill marks the sampled-margin variants
        for b, r in zip(bars, order):
            if r["stem"].endswith("_warm"):
                b.set_facecolor("white")
                b.set_edgecolor(FAM_COLOR[r["fam"]])
                b.set_linewidth(1.6)
                b.set_hatch("///")
        span = max(delta) if max(delta) > 0 else 1.0
        for yy, d, r in zip(y, delta, order):
            ax.text(d + 0.018 * span, yy, f"{r[key]:,.{dec}f}", va="center",
                    ha="left", fontsize=8.5)
        ax.set_xlim(0, span * 1.30)
        ax.set_title(f"{title}\n(0 = best model)", fontsize=10)
        ax.set_xlabel("difference from best")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.xaxis.set_major_formatter(FuncFormatter(_tickfmt))
        ax.grid(axis="x", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9.5)
    nd = {r["n_draws"] for r in order}
    foot = ("Bars show each model's shortfall from the best; the raw value is printed "
            "on the bar. Hatched = sampled margins, solid = fixed margins.")
    if len(nd) > 1:
        foot += ("  NOTE: posterior draws differ across models ("
                 + ", ".join(f"{r['name']}: {r['n_draws']}" for r in order
                             if r["n_draws"] != max(nd)) + f" vs {max(nd)}).")
    fig.suptitle("Faroe held-out model comparison, pooled test set (63 sites)",
                 fontsize=12.5)
    fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="0.3", wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(path, dpi=160)
    return path


def load_pointwise(outdir, rows, region="all"):
    """{stem: {metric: per-time array}} from the sidecars, if the run wrote them."""
    ptdir = os.path.join(outdir, "pointwise")
    out = {}
    for r in rows:
        fp = os.path.join(ptdir, f"{r['stem']}.{region}.npz")
        if os.path.exists(fp):
            with np.load(fp) as d:
                out[r["stem"]] = {k: d[k] for k in d.files}
    return out


def se_report(rows, pw, baseline=None):
    """Standard errors, and the PAIRED differences that are the real comparison.

    The scores are sums (log-score, WAIC) or means (CRPS) over T time replicates, so
    with the per-replicate values in hand:

        se(sum)  = sqrt(T) * sd(x_t)          se(mean) = sd(x_t) / sqrt(T)

    The time replicate is the exchangeable unit -- sites within a time share one latent
    field, so treating the n*T cells as independent would understate the error badly.

    The MARGINAL se of each model is nearly useless for ranking, because the models are
    scored on the SAME data and their per-replicate scores are strongly correlated: the
    difference of two noisy numbers is far less noisy than either. So the comparison
    reported here is the paired one, se(A - B) = sqrt(T) * sd(x_t^A - x_t^B), exactly as
    in the elpd-difference practice for LOO/WAIC. Treating the two as independent
    (sqrt(se_A^2 + se_B^2)) is the classic way to declare a real difference
    'not significant'.

    CAVEAT: this is the DATA (finite-test-set) error only. It does not include the Monte
    Carlo error of the posterior draws, and the replicates share one posterior sample,
    so they are not perfectly independent either. Read these as the right order of
    magnitude, not as exact."""
    if not pw:
        print("\nNo pointwise sidecars in compare/pointwise/ -- this run predates them,"
              "\nso standard errors cannot be formed from the saved output. Re-run "
              "task 6\n(a reduced --draws pass is enough for the SE) to generate them.")
        return
    # (sidecar key, heading, sum|mean, direction that is BETTER). The sidecar stores
    # elpd for WAIC, not WAIC itself, so higher is better there -- the sign flip WAIC
    # would need is exactly the kind of thing to get wrong silently, so it is named.
    keys = [("loglik", "log-score", "sum", "higher"),
            ("waic", "WAIC(elpd)", "sum", "higher"),
            ("twcrps_smooth", "TW-CRPS smooth", "mean", "lower")]
    have = [r for r in rows if r["stem"] in pw]
    base_row = next((r for r in have if r["stem"] == baseline), None)
    if baseline and base_row is None:
        print(f"\nnote: baseline {baseline!r} has no pointwise sidecar yet -- "
              f"falling back to the first available model.")
    print("\nSTANDARD ERRORS (data error over T time replicates)")
    for key, title, how, better in keys:
        avail = [r for r in have if key in pw[r["stem"]]]
        if not avail:
            continue
        ref = base_row if (base_row is not None and key in pw[base_row["stem"]]) \
            else avail[0]
        print(f"\n  {title}  ({better} = better)")
        for r in avail:
            x = np.asarray(pw[r["stem"]][key], float)
            T = x.size
            est = x.sum() if how == "sum" else x.mean()
            se = np.sqrt(T) * x.std(ddof=1) if how == "sum" else x.std(ddof=1) / np.sqrt(T)
            mark = "  <- baseline" if r is ref else ""
            print(f"    {r['name']:<24}{est:>14,.4f}  +- {se:>10,.4f}{mark}")
        others = [r for r in avail if r is not ref]
        if not others:
            continue
        print(f"    paired differences, MODEL - {ref['name']} "
              f"(|diff| > 2 se => distinguishable):")
        base = np.asarray(pw[ref["stem"]][key], float)
        for r in others:
            x = np.asarray(pw[r["stem"]][key], float)
            if x.size != base.size:
                print(f"      {r['name']:<24} (different T -- not comparable)")
                continue
            d = x - base                      # MODEL minus baseline
            T = d.size
            dd = d.sum() if how == "sum" else d.mean()
            sed = np.sqrt(T) * d.std(ddof=1) if how == "sum" else d.std(ddof=1) / np.sqrt(T)
            if abs(dd) <= 2 * sed:
                verdict = "within noise"
            else:
                wins = (dd > 0) if better == "higher" else (dd < 0)
                verdict = f"favours {r['name'] if wins else ref['name']}"
            print(f"      {r['name']:<24}{dd:>+14,.4f}  +- {sed:>9,.4f}   {verdict}")


# Reference model for the paired differences. Wang with FIXED margins is the sensible
# default: it is the two-stage treatment the pipeline actually settled on, so every
# other entry reads as "what changing one thing does relative to the production fit".
# The sampled-margin variants are not a neutral reference -- those chains fall into the
# known bad-margin trap (sigma ~3x too small, xi flipped positive), so differences
# measured against one of them confound the model with that pathology.
DEFAULT_BASELINE = "wang_fixmargins_stage1"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help=f"model stem the paired differences are measured against "
                         f"(default: {DEFAULT_BASELINE}). Use a CSV 'model' value.")
    args = ap.parse_args()
    outdir = os.path.join(F.OUT, "compare")
    rows = load_all_rows(os.path.join(outdir, "model_comparison.csv"))
    tex = os.path.join(outdir, "model_comparison_table.tex")
    png = os.path.join(outdir, "model_comparison_all.png")
    print(write_tex(rows, tex))
    make_figure(rows, png)
    print(f"\nWrote {tex}\n      {png}")
    # rankings, so a disagreement between criteria is visible rather than buried
    print("\nranking by each criterion (best first):")
    for key, title, _, lower, _ in METRICS:
        rk = sorted(rows, key=lambda r: (r[key] if lower else -r[key]))
        print(f"  {title:<22} " + "  >  ".join(r["name"].split(",")[0]
                                               + ("*" if r["stem"].endswith("_warm")
                                                  else "") for r in rk))
    print("  (* = sampled margins)")
    se_report(rows, load_pointwise(outdir, rows), baseline=args.baseline)


if __name__ == "__main__":
    main()
