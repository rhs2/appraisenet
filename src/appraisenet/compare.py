"""Statistical comparison and error profiling over the shared holdout.

Every model predicts the same holdout cars, so differences between models must be
judged on paired errors, never by comparing two separate scores by eye. A paired
bootstrap (resample holdout cars with replacement; apply the same draw to every
model) yields a 95% confidence interval for each model's error and for its gap to the
best model. A model whose gap interval includes zero is statistically tied with the
best; claiming a ranking between tied models would be overfitting the holdout.

Two error summaries are inferred, not one. MAPE is a mean and therefore a statement
about the tail; the median APE is a statement about the typical car. The corpus-scale
study separates the two: a design can hold the best median and the worst MAPE at the
same time, and a leaderboard that reports only one of them hides that.

`error_profiles` adds the distribution behind those two numbers: percentiles, the
share of cars inside the usual tolerances, the weight of the tail, and the same split
by price band, which is where the two summaries disagree most.

Everything here runs from the persisted `predictions/*.npz`, so it never refits
anything and can be applied to a finished study after the fact.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [(0, 10_000, "under $10k"), (10_000, 20_000, "$10k-20k"),
         (20_000, 40_000, "$20k-40k"), (40_000, np.inf, "over $40k")]


def _load_ape(out: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None]:
    """Per-model holdout APE (%) and signed log error on one shared, equal-length
    holdout, plus the holdout prices. The sign matters: a model can be wrong because it
    prices cars too high or because it prices them too low, and the two failures have
    different causes."""
    res_path, pred_dir = out / "results.csv", out / "predictions"
    if res_path.exists():
        order = pd.read_csv(res_path)["model"].tolist()
    elif pred_dir.exists():
        order = sorted(p.stem for p in pred_dir.glob("*.npz"))
    else:
        return {}, {}, None
    ape: dict[str, np.ndarray] = {}
    signed: dict[str, np.ndarray] = {}
    price = None
    for name in order:
        npz = pred_dir / f"{name}.npz"
        if npz.exists():
            arr = np.load(npz)
            err = arr["holdout_pred"].astype(np.float64) - arr["holdout_y"]
            ape[name] = np.abs(np.exp(err) - 1) * 100
            signed[name] = err
            if "holdout_price" in arr.files:
                price = arr["holdout_price"]
    if not ape:
        return {}, {}, None
    # pairing requires one shared holdout; keep the majority length, drop strays
    n = Counter(len(v) for v in ape.values()).most_common(1)[0][0]
    ape = {k: v for k, v in ape.items() if len(v) == n}
    signed = {k: v for k, v in signed.items() if k in ape}
    if price is not None and len(price) != n:
        price = None
    return ape, signed, price


def _paired_bootstrap(ape: dict[str, np.ndarray], n_boot: int, seed: int
                      ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Bootstrap distributions of the mean and median APE under shared resamples.

    Draws are generated one at a time on purpose: a corpus-scale holdout times 4,000
    draws is a multi-gigabyte index matrix if it is materialised in one array.
    """
    names = list(ape)
    n = len(ape[names[0]])
    rng = np.random.RandomState(seed)
    means = {k: np.empty(n_boot) for k in names}
    medians = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        for k in names:
            v = ape[k][idx]
            means[k][b] = v.mean()
            medians[k][b] = np.median(v)
    return means, medians


def comparison_stats(out: Path, n_boot: int = 4000, seed: int = 42) -> pd.DataFrame | None:
    """Write comparison_stats.csv next to results.csv; None when artifacts are missing."""
    out = Path(out)
    ape, _, _ = _load_ape(out)
    if not ape:
        return None
    means, medians = _paired_bootstrap(ape, n_boot, seed)

    champion = min(ape, key=lambda k: float(ape[k].mean()))
    best_median = min(ape, key=lambda k: float(np.median(ape[k])))

    def interval(dist: np.ndarray, places: int = 3) -> tuple[float, float]:
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return round(float(lo), places), round(float(hi), places)

    rows = []
    for name, e in ape.items():
        d_lo, d_hi = interval(means[name] - means[champion])
        m_lo, m_hi = interval(medians[name] - medians[best_median])
        ci_lo, ci_hi = interval(means[name])
        mci_lo, mci_hi = interval(medians[name])
        rows.append({"model": name,
                     "holdout_mape_pct": round(float(e.mean()), 3),
                     "mape_ci95_low": ci_lo, "mape_ci95_high": ci_hi,
                     "delta_vs_champion": round(float(e.mean() - ape[champion].mean()), 3),
                     "delta_ci95_low": d_lo, "delta_ci95_high": d_hi,
                     "tied_with_champion": bool(name == champion or d_lo <= 0),
                     "holdout_median_ape_pct": round(float(np.median(e)), 3),
                     "median_ci95_low": mci_lo, "median_ci95_high": mci_hi,
                     "median_delta_vs_best": round(float(np.median(e) - np.median(ape[best_median])), 3),
                     "median_delta_ci95_low": m_lo, "median_delta_ci95_high": m_hi,
                     "tied_with_best_median": bool(name == best_median or m_lo <= 0)})
    df = pd.DataFrame(rows).sort_values("holdout_mape_pct").reset_index(drop=True)
    df.to_csv(out / "comparison_stats.csv", index=False)

    # every pair, not only the gap to the champion: a claim about two models in the
    # middle of the table needs its own paired interval, and the draws are already here
    order = df["model"].tolist()
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            d = means[a] - means[b]
            lo, hi = interval(d, 4)
            pairs.append({"model_a": a, "model_b": b,
                          "delta_mape_pct": round(float(ape[a].mean() - ape[b].mean()), 4),
                          "delta_ci95_low": lo, "delta_ci95_high": hi,
                          "separated": bool(lo > 0 or hi < 0)})
    pd.DataFrame(pairs).to_csv(out / "pairwise_mape.csv", index=False)
    return df


def error_profiles(out: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Write error_profile.csv (distribution shape) and price_bands.csv (band x model).

    MAPE compresses a whole error distribution into one number that the tail dominates.
    These two tables keep the distribution: where the bulk of the cars sit, how heavy
    the tail is, and how much of the total error the tail carries.
    """
    out = Path(out)
    ape, signed, price = _load_ape(out)
    if not ape:
        return None, None
    rows = []
    for name, e in ape.items():
        tail = e > 25
        big = e > 50
        over = float((signed[name][big] > 0).mean() * 100) if big.any() else float("nan")
        rows.append({"model": name,
                     "mape_pct": round(float(e.mean()), 3),
                     "p50": round(float(np.median(e)), 3),
                     "p75": round(float(np.percentile(e, 75)), 3),
                     "p90": round(float(np.percentile(e, 90)), 3),
                     "p95": round(float(np.percentile(e, 95)), 3),
                     "p99": round(float(np.percentile(e, 99)), 3),
                     "within_5_pct": round(float((e <= 5).mean() * 100), 3),
                     "within_10_pct": round(float((e <= 10).mean() * 100), 3),
                     "within_20_pct": round(float((e <= 20).mean() * 100), 3),
                     "over_25_pct": round(float(tail.mean() * 100), 3),
                     "over_50_pct": round(float((e > 50).mean() * 100), 3),
                     "over_100_pct": round(float((e > 100).mean() * 100), 3),
                     "over_50_overpriced_pct": round(over, 3),
                     "tail_share_of_error_pct": round(float(e[tail].sum() / e.sum() * 100), 3)})
    profile = pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)
    profile.to_csv(out / "error_profile.csv", index=False)

    bands = None
    if price is not None:
        brows = []
        for lo, hi, label in BANDS:
            mask = (price >= lo) & (price < hi)
            if not mask.any():
                continue
            for name in profile["model"]:
                e = ape[name][mask]
                brows.append({"band": label, "n": int(mask.sum()), "model": name,
                              "mape_pct": round(float(e.mean()), 3),
                              "median_ape_pct": round(float(np.median(e)), 3)})
        bands = pd.DataFrame(brows)
        bands.to_csv(out / "price_bands.csv", index=False)
    return profile, bands
