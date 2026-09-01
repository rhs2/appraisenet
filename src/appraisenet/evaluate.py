"""Figures, regenerated entirely from the persisted run artifacts (results.csv,
segments.csv, error_profile.csv, price_bands.csv, predictions/*.npz) so
`appraisenet report` can rebuild the README and paper graphics without re-running any
model. Colours are one fixed hue per model family (Okabe-Ito, colour-blind safe), so a
family keeps its colour across every figure and across both tiers of the study."""
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
                  "boosting": "#E69F00", "deep tabular": "#CC79A7", "hybrid": "#D55E00",
                  "anchored": "#B8860B"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def make_figures(out: Path) -> list[Path]:
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    results = pd.read_csv(out / "results.csv")
    made = [_comparison(results, fig_dir), _coverage(results, fig_dir),
            _cost_accuracy(results, fig_dir)]
    seg_path = out / "segments.csv"
    if seg_path.exists():
        made.append(_segments(pd.read_csv(seg_path), fig_dir))
    prof_path = out / "error_profile.csv"
    if prof_path.exists():
        made.append(_error_profile(pd.read_csv(prof_path), results, fig_dir))
    band_path = out / "price_bands.csv"
    if band_path.exists():
        made.append(_price_bands(pd.read_csv(band_path), results, fig_dir))
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
            tied = set(st.index[st["tied_with_champion"].fillna(False)]) - {res.iloc[0]["model"]}
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(r) + 1.2))
    colours = [FAMILY_COLOURS.get(f, "#999999") for f in r["family"]]
    ax.barh(r["model"], r["holdout_mape_pct"], color=colours, height=0.62,
            xerr=xerr, error_kw={"lw": 0.9, "capsize": 2, "ecolor": "#333333"})
    ax.scatter(r["cv_mape_pct"], np.arange(len(r)), marker="|", s=180, color="#111111",
               label="cross-validation", zorder=3)
    # anchor annotations past the error bar and the CV tick, whichever reaches further
    reach = r["holdout_mape_pct"].values if xerr is None else r["holdout_mape_pct"].values + xerr[1]
    anchors = np.maximum(reach, r["cv_mape_pct"].values)
    for i, (m, v, a, w) in enumerate(zip(r["model"], r["holdout_mape_pct"], anchors,
                                         r["holdout_within_10_pct"])):
        mark = " †" if m in tied else ""
        ax.annotate(f"{v:.2f}%{mark}  ({w:.0f}% within 10%)", (a, i), xytext=(5, -3),
                    textcoords="offset points", fontsize=8)
    ax.set_xlim(0, float(anchors.max()) * 1.36)
    xlabel = "holdout MAPE (%), lower is better"
    if xerr is not None:
        xlabel += "; error bars: 95% paired CI; † tied with the champion"
    ax.set_xlabel(xlabel)
    ax.set_title("Model families under the AppraiseNet Evaluation Protocol")
    present = [f for f in FAMILY_COLOURS if f in set(r["family"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOURS[f]) for f in present]
    ax.legend(handles + [plt.Line2D([], [], color="#111111", marker="|", ls="", ms=12)],
              present + ["cross-validation"], fontsize=7.5,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    p = fig_dir / "comparison.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _coverage(res: pd.DataFrame, fig_dir: Path) -> Path:
    """Dot plot: coverage barely varies (that is the guarantee working), width is the story."""
    r = res.sort_values("holdout_width_pct_of_price", ascending=False)
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(r) + 1.3))
    for i, (_, row) in enumerate(r.iterrows()):
        ax.scatter(row["holdout_width_pct_of_price"], i, s=64, zorder=3,
                   color=FAMILY_COLOURS.get(row["family"], "#999999"))
        ax.annotate(f"{row['holdout_width_pct_of_price']:.0f}% wide at "
                    f"{row['holdout_coverage_pct']:.1f}% coverage",
                    (row["holdout_width_pct_of_price"], i), xytext=(8, -3),
                    textcoords="offset points", fontsize=7.5)
    ax.set_yticks(np.arange(len(r)), r["model"], fontsize=8)
    ax.set_xlim(0, float(r["holdout_width_pct_of_price"].max()) * 1.5)
    cov = res["holdout_coverage_pct"]
    ax.annotate(f"holdout coverage stays {cov.min():.1f}-{cov.max():.1f}% vs the 80% target,\n"
                "for every model: the conformal guarantee at work",
                (0.02, 0.03), xycoords="axes fraction", fontsize=8, color="#666666")
    ax.set_xlabel("median 80%-interval width (% of price), narrower is better")
    ax.set_title("Split-conformal intervals: width differs, coverage holds", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "coverage_width.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _calibration(name: str, arr, fig_dir: Path) -> Path:
    pred, y = np.exp(arr["holdout_pred"]), arr["holdout_price"]
    dec = pd.qcut(pred, 10, duplicates="drop")
    g = pd.DataFrame({"pred": pred, "y": y, "dec": dec}).groupby("dec", observed=True).mean()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = [min(g["pred"].min(), g["y"].min()) * 0.9, max(g["pred"].max(), g["y"].max()) * 1.06]
    ax.plot(lim, lim, color="#999999", lw=1, ls="--")
    ax.plot(g["pred"], g["y"], marker="o", color="#E69F00", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [t for t in (2_000, 5_000, 10_000, 20_000, 40_000, 80_000) if lim[0] <= t <= lim[1]]
    labels = [f"${t // 1000}k" for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("mean predicted price (decile)")
    ax.set_ylabel("mean actual price")
    ax.set_title("Champion calibration by predicted-price decile", fontsize=10)
    ax.annotate(name, (0.04, 0.94), xycoords="axes fraction", fontsize=8.5, color="#666666")
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
    """Top models by MAPE plus the best model by MEDIAN error, whose curve crosses them:
    ahead on the typical car, behind in the tail. One number cannot show that."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    top = res.head(3)["model"].tolist()
    if "holdout_median_ape_pct" in res.columns:
        best_med = res.sort_values("holdout_median_ape_pct").iloc[0]["model"]
        top += [best_med] if best_med not in top else [res.iloc[3]["model"]]
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


def _error_profile(prof: pd.DataFrame, res: pd.DataFrame, fig_dir: Path) -> Path:
    """The two summaries side by side. Left: the gap between the typical car (median APE)
    and the average car (MAPE), which is the weight of the tail. Right: how often a model
    misses by more than half the price, which is what a buyer would call a wrong answer."""
    fam = res.set_index("model")["family"]
    p = prof.sort_values("p50", ascending=False)   # best at the top, as in every other figure
    colours = [FAMILY_COLOURS.get(fam.get(m, ""), "#999999") for m in p["model"]]
    y = np.arange(len(p))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 0.42 * len(p) + 1.6),
                                 gridspec_kw={"width_ratios": [1.65, 1]})
    a1.hlines(y, p["p50"], p["mape_pct"], color="#BBBBBB", lw=2.4, zorder=1)
    a1.scatter(p["p50"], y, s=52, color=colours, zorder=3)
    a1.scatter(p["mape_pct"], y, s=52, color=colours, zorder=3, marker="D",
               edgecolors="#FFFFFF", linewidths=0.8)
    for i, (m, lo, hi) in enumerate(zip(p["model"], p["p50"], p["mape_pct"])):
        a1.annotate(f"{hi - lo:+.2f}", (hi, i), xytext=(9, -3), textcoords="offset points",
                    fontsize=7.5, color="#555555")
    a1.set_yticks(y, p["model"], fontsize=8)
    a1.set_xlim(0, float(p["mape_pct"].max()) * 1.2)
    a1.set_xlabel("holdout error (%)")
    a1.legend([plt.Line2D([], [], color="#555555", marker="o", ls="", ms=7),
               plt.Line2D([], [], color="#555555", marker="D", ls="", ms=7)],
              ["median APE (typical car)", "MAPE (mean, tail-driven)"], fontsize=7.5,
              loc="lower left")
    a1.set_title("Typical car vs average car", fontsize=10)

    a2.barh(y, p["over_50_pct"], color=colours, height=0.6)
    for i, v in enumerate(p["over_50_pct"]):
        a2.annotate(f"{v:.2f}%", (v, i), xytext=(4, -3), textcoords="offset points", fontsize=7.5)
    a2.set_yticks(y, [""] * len(p))
    a2.set_xlim(0, float(p["over_50_pct"].max()) * 1.35)
    a2.set_xlabel("share of holdout cars missed by more than 50%")
    a2.set_title("The tail", fontsize=10)
    fig.tight_layout()
    out = fig_dir / "error_profile.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _price_bands(bands: pd.DataFrame, res: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Where the ranking flips. Left: absolute accuracy per band, which is dominated by the
    cheap end. Right: the same models as a gap to the champion, which is the only way to see
    that the anchored residual design wins the two upper bands while losing the corpus mean."""
    from .compare import BANDS
    order = [label for _, _, label in BANDS if label in set(bands["band"])]
    champion = res.iloc[0]["model"]
    picks = [champion]
    for name in ("anchored_lgbm", "anchored_blend", "ft_transformer", "lightgbm"):
        if name in set(bands["model"]) and name not in picks and len(picks) < 4:
            picks.append(name)
    if len(picks) < 2 or not order:
        return None
    palette = ["#E69F00", "#0072B2", "#B8860B", "#CC79A7"]
    x = np.arange(len(order))
    by = {m: bands[bands["model"] == m].set_index("band").reindex(order)["mape_pct"] for m in picks}
    gaps = {m: (by[m] - by[champion]).values for m in picks}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for name, colour in zip(picks, palette):
        a1.plot(x, by[name], marker="o", ms=5, lw=1.8, color=colour, label=name)
        a2.plot(x, gaps[name], marker="o", ms=5, lw=1.8, color=colour, clip_on=True)
    a2.axhline(0, color="#666666", lw=1, ls=":")
    # one band can miss by ten points while the rest differ by hundredths: keep the
    # readable range and label whatever leaves the frame instead of flattening everything
    inside = np.concatenate([v[np.abs(v) <= 2] for v in gaps.values()])
    if inside.size:
        lo, hi = float(inside.min()) - 0.2, float(inside.max()) + 0.25
        a2.set_ylim(lo, hi)
        for name, colour in zip(picks, palette):
            for xi, v in zip(x, gaps[name]):
                if v > hi or v < lo:
                    edge = hi if v > hi else lo
                    a2.annotate(f"{v:+.1f}", (xi, edge), xytext=(0, -12 if v > hi else 10),
                                textcoords="offset points", ha="center", fontsize=8,
                                color=colour, fontweight="bold")
    n_by_band = bands.drop_duplicates("band").set_index("band")["n"].reindex(order)
    labels = [f"{b}\n(n={int(n):,})" for b, n in zip(order, n_by_band)]
    for ax in (a1, a2):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlim(-0.3, len(order) - 0.7)
    a1.set_ylabel("holdout MAPE (%), lower is better")
    a1.set_title("Accuracy by price band", fontsize=10)
    a1.legend(fontsize=8)
    a2.set_ylabel(f"MAPE minus {champion} (points)")
    a2.set_title("The same models, as a gap to the champion", fontsize=10)
    a2.annotate("below the dotted line: better than the champion", (0.97, 0.05),
                xycoords="axes fraction", fontsize=8, color="#666666", ha="right")
    fig.suptitle("The cheap end decides the corpus mean", fontsize=11)
    fig.tight_layout()
    out = fig_dir / "price_bands.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _cost_accuracy(res: pd.DataFrame, fig_dir: Path) -> Path | None:
    """What the last hundredths of a MAPE point cost in compute. The protocol total is
    five folds plus the final fit, so it is directly comparable across models."""
    if "fit_seconds" not in res.columns or res["fit_seconds"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ordered = res.sort_values("fit_seconds").reset_index(drop=True)
    for i, r in ordered.iterrows():
        ax.scatter(r["fit_seconds"] / 60, r["holdout_mape_pct"], s=64, zorder=3,
                   color=FAMILY_COLOURS.get(r["family"], "#999999"))
        # models cluster inside a one-point MAPE range: stagger the labels over three
        # rows so neighbours in compute cost do not print on top of each other
        ax.annotate(r["model"], (r["fit_seconds"] / 60, r["holdout_mape_pct"]),
                    xytext=[(8, 5), (8, -6), (8, -16)][i % 3],
                    textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("protocol fit time (minutes, log scale)")
    ax.set_ylabel("holdout MAPE (%), lower is better")
    ax.set_title("Accuracy against compute", fontsize=10)
    present = [f for f in FAMILY_COLOURS if f in set(res["family"])]
    ax.legend([plt.Line2D([], [], color=FAMILY_COLOURS[f], marker="o", ls="", ms=7) for f in present],
              present, fontsize=7.5, loc="upper right")
    ax.set_xlim(res["fit_seconds"].min() / 60 * 0.45, res["fit_seconds"].max() / 60 * 3.4)
    lo, hi = res["holdout_mape_pct"].min(), res["holdout_mape_pct"].max()
    ax.set_ylim(lo - (hi - lo) * 0.20, hi + (hi - lo) * 0.10)
    fig.tight_layout()
    out = fig_dir / "cost_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out
