"""Statistical comparison over the shared holdout.

Every model predicts the same holdout cars, so differences between models must be
judged on paired errors, never by comparing two separate scores by eye. A paired
bootstrap (resample holdout cars with replacement; apply the same draw to every
model) yields a 95% confidence interval for each model's MAPE and for its gap to
the champion. A model whose gap interval includes zero is statistically tied with
the best; claiming a ranking between tied models would be overfitting the holdout.

Runs entirely from the persisted `predictions/*.npz`, so it never refits anything
and can be applied to a finished study after the fact.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def comparison_stats(out: Path, n_boot: int = 4000, seed: int = 42) -> pd.DataFrame | None:
    """Write comparison_stats.csv next to results.csv; None when artifacts are missing."""
    out = Path(out)
    res_path, pred_dir = out / "results.csv", out / "predictions"
    if res_path.exists():
        order = pd.read_csv(res_path)["model"].tolist()
    elif pred_dir.exists():
        order = sorted(p.stem for p in pred_dir.glob("*.npz"))
    else:
        return None
    ape: dict[str, np.ndarray] = {}
    for name in order:
        npz = pred_dir / f"{name}.npz"
        if npz.exists():
            arr = np.load(npz)
            ape[name] = np.abs(np.exp(arr["holdout_pred"] - arr["holdout_y"]) - 1) * 100
    if not ape:
        return None
    # pairing requires one shared holdout; keep the majority length, drop strays
    n = Counter(len(v) for v in ape.values()).most_common(1)[0][0]
    ape = {k: v for k, v in ape.items() if len(v) == n}

    champion = min(ape, key=lambda k: float(ape[k].mean()))
    idx = np.random.RandomState(seed).randint(0, n, size=(n_boot, n))
    boot = {k: v[idx].mean(axis=1) for k, v in ape.items()}

    rows = []
    for name in (m for m in order if m in ape):
        delta = boot[name] - boot[champion]
        d_lo, d_hi = (float(x) for x in np.percentile(delta, [2.5, 97.5]))
        rows.append({"model": name,
                     "holdout_mape_pct": round(float(ape[name].mean()), 3),
                     "mape_ci95_low": round(float(np.percentile(boot[name], 2.5)), 3),
                     "mape_ci95_high": round(float(np.percentile(boot[name], 97.5)), 3),
                     "delta_vs_champion": round(float(ape[name].mean() - ape[champion].mean()), 3),
                     "delta_ci95_low": round(d_lo, 3),
                     "delta_ci95_high": round(d_hi, 3),
                     "tied_with_champion": bool(name == champion or d_lo <= 0)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison_stats.csv", index=False)
    return df
