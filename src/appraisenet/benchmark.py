"""Benchmark orchestrator: run the zoo under the protocol, persist everything needed to
rebuild every table and figure (results.csv, per-model prediction arrays, model cards),
and track each run in MLflow (JSONL fallback)."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import comparison_stats
from .config import Config
from .data import TARGET, engineer, load_listings, protocol_split
from .models import zoo
from .protocol import cqr_champion, run_model, segment_table
from .tracking import Tracker


def run_benchmark(cfg: Config) -> pd.DataFrame:
    df, synthetic = load_listings(max_rows=cfg.max_rows, seed=cfg.protocol.seed)
    df = engineer(df, cfg.protocol)
    split = protocol_split(df, cfg.protocol)
    split.synthetic = synthetic
    names = cfg.models or zoo.ZOO

    out = cfg.out_dir
    (out / "predictions").mkdir(exist_ok=True)
    tracker = Tracker(experiment=f"appraisenet-{cfg.name}", reports_dir=out)
    print(f"data: {len(df):,} listings ({'SYNTHETIC' if synthetic else 'private'}) | "
          f"train {len(split.train):,} / holdout {len(split.holdout):,} | models: {len(names)} | tracking: {tracker.backend}")

    rows, hold_preds = [], {}
    for name in names:
        res = run_model(name, split, cfg)
        rows.append(res.row())
        hold_preds[name] = res.holdout_pred
        np.savez_compressed(out / "predictions" / f"{name}.npz",
                            oof=res.oof, holdout_pred=res.holdout_pred,
                            holdout_y=split.holdout[TARGET].values,
                            holdout_price=split.holdout["price"].values)
        with tracker.run(name, tags={"family": res.family, "synthetic": str(synthetic)}) as run:
            run.params({"model": name, "folds": cfg.protocol.folds, "rows": len(df)})
            run.metrics({**{f"cv_{k}": v for k, v in res.cv.items()},
                         **{f"holdout_{k}": v for k, v in res.holdout.items()},
                         **res.interval, "fit_seconds": res.fit_seconds})
        print(f"  {name:22s} CV MAPE {res.cv['mape_pct']:6.2f}%  holdout {res.holdout['mape_pct']:6.2f}%  "
              f"coverage {res.interval['holdout_coverage_pct']:5.1f}%  ({res.fit_seconds:.0f}s)", flush=True)

    results = pd.DataFrame(rows).sort_values("holdout_mape_pct").reset_index(drop=True)
    results.to_csv(out / "results.csv", index=False)
    comparison_stats(out)

    extras = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rows": len(df), "train": len(split.train), "holdout": len(split.holdout),
              "synthetic": bool(synthetic), "config": cfg.name, "platform": platform.platform()}
    if "lightgbm" in names and not synthetic:
        extras["cqr"] = cqr_champion(split, cfg)
    seg = segment_table(split.holdout, {n: hold_preds[n] for n in results["model"].head(5)})
    seg.to_csv(out / "segments.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(extras, indent=1))
    _model_cards(results, extras, out)

    from .evaluate import make_figures
    make_figures(out)
    return results


def _model_cards(results: pd.DataFrame, extras: dict, out: Path) -> None:
    cards = out / "model_cards"
    cards.mkdir(exist_ok=True)
    for _, r in results.iterrows():
        cards.joinpath(f"{r['model']}.md").write_text(f"""# Model card: {r['model']}

- family: {r['family']}
- protocol: {extras['rows']:,} listings, {extras['train']:,} train / {extras['holdout']:,} holdout,
  5-fold out-of-fold selection, holdout scored once ({'synthetic corpus' if extras['synthetic'] else 'private dataset'})
- cross-validation: MAPE {r['cv_mape_pct']}%, median APE {r['cv_median_ape_pct']}%, R2(log) {r['cv_r2_log']}
- holdout: MAPE {r['holdout_mape_pct']}%, median APE {r['holdout_median_ape_pct']}%,
  R2(log) {r['holdout_r2_log']}, within 10%: {r['holdout_within_10_pct']}%
- 80% conformal interval on the holdout: coverage {r['holdout_coverage_pct']}%,
  median width {r['holdout_width_pct_of_price']}% of price
- fit time (protocol total): {r['fit_seconds']} s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, ${f'{2000:,}'}-${f'{100000:,}'} price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
""")
