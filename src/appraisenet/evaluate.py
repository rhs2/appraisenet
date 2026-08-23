"""Figures, regenerated entirely from the persisted run artifacts (results.csv,
segments.csv, predictions/*.npz) so `appraisenet report` can rebuild the README and
paper graphics without re-running any model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colour-blind safe
FAMILY_COLOURS = {"linear": "#0072B2", "instance": "#56B4E9", "bagged trees": "#009E73",
                  "boosting": "#E69F00", "deep tabular": "#CC79A7", "hybrid": "#D55E00"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def make_figures(out: Path) -> list[Path]:
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    results = pd.read_csv(out / "results.csv")
    made = [_comparison(results, fig_dir), _coverage(results, fig_dir)]
    seg_path = out / "segments.csv"
    if seg_path.exists():
        made.append(_segments(pd.read_csv(seg_path), fig_dir))
    champion = results.iloc[0]["model"]
    npz = out / "predictions" / f"{champion}.npz"
    if npz.exists():
        made += [_calibration(champion, np.load(npz), fig_dir), _ecdf(results, out, fig_dir)]
    meta = out / "run_meta.json"
    if meta.exists():
        print("figures for run:", json.loads(meta.read_text()).get("generated"))
    return [p for p in made if p]


def _comparison(res: pd.DataFrame, fig_dir: Path) -> Path:
    r = res.sort_values("holdout_mape_pct", ascending=False)
    stats_path = fig_dir.parent / "comparison_stats.csv"
    xerr, tied = None, set()
    if stats_path.exists():
        st = pd.read_csv(stats_path).set_index("model").reindex(r["model"])
        if st["mape_ci95_low"].notna().all():
            xerr = np.vstack([(st["holdout_mape_pct"] - st["mape_ci95_low"]).values,
                              (st["mape_ci95_high"] - st["holdout_mape_pct"]).values])
            tied = set(st.index[st["tied_with_champion"].fillna(False)])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(r) + 1.2))
    colours = [FAMILY_COLOURS.get(f, "#999999") for f in r["family"]]
    ax.barh(r["model"], r["holdout_mape_pct"], color=colours, height=0.62,
            xerr=xerr, error_kw={"lw": 0.9, "capsize": 2, "ecolor": "#333333"})
    ax.scatter(r["cv_mape_pct"], np.arange(len(r)), marker="|", s=180, color="#111111",
               label="cross-validation", zorder=3)
    for i, (m, v, w) in enumerate(zip(r["model"], r["holdout_mape_pct"], r["holdout_within_10_pct"])):
        mark = " †" if m in tied else ""
        ax.annotate(f"{v:.2f}%{mark}  ({w:.0f}% within 10%)", (v, i), xytext=(5, -3),
                    textcoords="offset points", fontsize=8)
    xlabel = "holdout MAPE (%), lower is better"
    if xerr is not None:
        xlabel += "; error bars: 95% paired-bootstrap CI, † statistically tied with the champion"
    ax.set_xlabel(xlabel)
    ax.set_title("AppraiseNet: model families under one leakage-free protocol")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOURS.values()]
    ax.legend(handles + [plt.Line2D([], [], color="#111111", marker="|", ls="", ms=12)],
              list(FAMILY_COLOURS) + ["cross-validation"], fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    p = fig_dir / "comparison.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _coverage(res: pd.DataFrame, fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for _, r in res.iterrows():
        ax.scatter(r["holdout_width_pct_of_price"], r["holdout_coverage_pct"],
                   color=FAMILY_COLOURS.get(r["family"], "#999999"), s=42, zorder=3)
        ax.annotate(r["model"], (r["holdout_width_pct_of_price"], r["holdout_coverage_pct"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.axhline(80, color="#666666", ls="--", lw=1)
    ax.annotate("80% target", (ax.get_xlim()[0], 80), xytext=(4, 4), textcoords="offset points",
                fontsize=8, color="#666666")
    ax.set_xlabel("median interval width (% of price), narrower is better")
    ax.set_ylabel("holdout coverage (%)")
    ax.set_title("Split-conformal 80% intervals: every model, honest coverage")
    fig.tight_layout()
    p = fig_dir / "coverage_width.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _calibration(name: str, arr, fig_dir: Path) -> Path:
    pred, y = np.exp(arr["holdout_pred"]), arr["holdout_price"]
    dec = pd.qcut(pred, 10, duplicates="drop")
    g = pd.DataFrame({"pred": pred, "y": y, "dec": dec}).groupby("dec", observed=True).mean()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = [min(g["pred"].min(), g["y"].min()) * 0.9, max(g["pred"].max(), g["y"].max()) * 1.06]
    ax.plot(lim, lim, color="#999999", lw=1, ls="--")
    ax.plot(g["pred"], g["y"], marker="o", color="#E69F00", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted price (decile, $)")
    ax.set_ylabel("mean actual price ($)")
    ax.set_title(f"Champion calibration by predicted-price decile ({name})")
    fig.tight_layout()
    p = fig_dir / "calibration.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _segments(seg: pd.DataFrame, fig_dir: Path) -> Path:
    models = [c for c in seg.columns if c not in ("segment", "n")]
    M = seg[models].values
    fig, ax = plt.subplots(figsize=(1.35 * len(models) + 2.4, 0.4 * len(seg) + 1.4))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(models)), models, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(seg)), [f"{s}  (n={n})" for s, n in zip(seg["segment"], seg["n"])], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="#111111" if M[i, j] < np.nanmax(M) * 0.7 else "#ffffff")
    ax.set_title("Holdout MAPE (%) by segment: where pricing models hide their weaknesses")
    fig.colorbar(im, shrink=0.75)
    ax.grid(False)
    fig.tight_layout()
    p = fig_dir / "segments.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _ecdf(res: pd.DataFrame, out: Path, fig_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    top = res.head(4)["model"].tolist()
    palette = ["#E69F00", "#0072B2", "#009E73", "#CC79A7"]
    for name, colour in zip(top, palette):
        npz = out / "predictions" / f"{name}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)
        ape = np.sort(np.abs(np.exp(arr["holdout_pred"] - arr["holdout_y"]) - 1)) * 100
        ax.plot(ape, np.arange(1, len(ape) + 1) / len(ape) * 100, label=name, color=colour, lw=1.6)
    ax.set_xlim(0, 40)
    ax.axvline(10, color="#666666", ls=":", lw=1)
    ax.set_xlabel("absolute percentage error (%)")
    ax.set_ylabel("share of holdout cars within (%)")
    ax.set_title("Error distribution, top models (holdout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "error_ecdf.png"
    fig.savefig(p)
    plt.close(fig)
    return p
