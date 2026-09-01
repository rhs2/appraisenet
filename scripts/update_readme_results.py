"""Regenerate the results section of README.md from reports/ artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    res = pd.read_csv(ROOT / "reports/results.csv")
    meta = json.loads((ROOT / "reports/run_meta.json").read_text())
    seg = pd.read_csv(ROOT / "reports/segments.csv")
    stats_path = ROOT / "reports/comparison_stats.csv"
    stats = pd.read_csv(stats_path).set_index("model") if stats_path.exists() else None

    prof_path = ROOT / "reports/error_profile.csv"
    prof = pd.read_csv(prof_path) if prof_path.exists() else None
    band_path = ROOT / "reports/price_bands.csv"
    bands = pd.read_csv(band_path) if band_path.exists() else None
    pair_path = ROOT / "reports/pairwise_mape.csv"
    pairs = pd.read_csv(pair_path) if pair_path.exists() else None

    top = res.iloc[0]
    lines = ["## Results", ""]
    corpus = "synthetic corpus (illustrative only)" if meta["synthetic"] else \
        f"{meta['rows']:,} real listings ({meta['train']:,} train / {meta['holdout']:,} holdout)"
    lines.append(f"**{corpus}**, protocol as below; champion: **{top['model']}** at "
                 f"**{top['holdout_mape_pct']}% holdout MAPE** "
                 f"(median {top['holdout_median_ape_pct']}%, {top['holdout_within_10_pct']}% of cars within 10%).")
    best_med = res.sort_values("holdout_median_ape_pct").iloc[0]
    if best_med["model"] != top["model"]:
        lines.append("")
        lines.append(f"The leaderboard is not a total order: by **median** APE the best model is "
                     f"**{best_med['model']}** ({best_med['holdout_median_ape_pct']}% against "
                     f"{top['holdout_median_ape_pct']}%), which is also the worst model by MAPE "
                     f"({best_med['holdout_mape_pct']}%). MAPE is a mean and belongs to the tail; "
                     "the error profile below separates the two.")
    lines += ["", "| model | family | CV MAPE | holdout MAPE | 95% CI | median APE | within 10% | 80% interval coverage | width (% of price) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        name, ci = r["model"], ""
        if stats is not None and r["model"] in stats.index:
            s = stats.loc[r["model"]]
            ci = f"{s['mape_ci95_low']}-{s['mape_ci95_high']}%"
            if bool(s["tied_with_champion"]) and r["model"] != top["model"]:
                name = f"{name} †"
        lines.append(f"| {name} | {r['family']} | {r['cv_mape_pct']}% | **{r['holdout_mape_pct']}%** | {ci} | "
                     f"{r['holdout_median_ape_pct']}% | {r['holdout_within_10_pct']}% | "
                     f"{r['holdout_coverage_pct']}% | {r['holdout_width_pct_of_price']}% |")
    if stats is not None:
        tied_others = [m for m in stats.index
                       if bool(stats.loc[m, "tied_with_champion"]) and m != top["model"]]
        if tied_others:
            lines += ["", "† statistically tied with the champion: the 95% paired-bootstrap interval of "
                          "the MAPE gap (4,000 resamples of the shared holdout) includes zero."]
        else:
            lines += ["", "The champion's lead is statistically significant: no other model's 95% "
                          "paired-bootstrap MAPE-gap interval includes zero."]
    if "cqr" in meta:
        c = meta["cqr"]
        lines += ["", f"Production interval (conformalised quantile regression on the production "
                      f"configuration): **{c['cqr_holdout_coverage_pct']}% coverage** "
                      f"at a median width of {c['cqr_holdout_width_pct_of_price']}% of price."]
    if pairs is not None:
        tied = int((~pairs["separated"]).sum())
        lines += ["", f"Across all {len(pairs)} model pairs, {tied} are statistically tied on the "
                      "shared holdout (95% paired-bootstrap interval of the gap includes zero); "
                      "`reports/pairwise_mape.csv` has every pair."]
    if prof is not None:
        lines += ["", "**The distribution behind the leaderboard** (`reports/error_profile.csv`):", "",
                  "| model | median APE | MAPE | within 5% | p90 | p99 | miss > 50% | share of total error from misses > 25% |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in prof.iterrows():
            lines.append(f"| {r['model']} | **{r['p50']}%** | {r['mape_pct']}% | {r['within_5_pct']}% | "
                         f"{r['p90']}% | {r['p99']}% | {r['over_50_pct']}% | {r['tail_share_of_error_pct']}% |")
    if bands is not None:
        wide = bands.pivot(index="band", columns="model", values="mape_pct")
        n_by = bands.drop_duplicates("band").set_index("band")["n"]
        cols = [m for m in dict.fromkeys([top["model"], "anchored_lgbm", best_med["model"],
                                          "ft_transformer"]) if m in wide.columns]
        order = [b for b in ("under $10k", "$10k-20k", "$20k-40k", "over $40k") if b in wide.index]
        lines += ["", "**Holdout MAPE by price band** (`reports/price_bands.csv`):", "",
                  "| band | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for b in order:
            lines.append(f"| {b} | {int(n_by[b]):,} | "
                         + " | ".join(f"{wide.loc[b, m]}%" for m in cols) + " |")
    lines += ["", "**Holdout MAPE by segment (top models):**", ""]
    cols = [c for c in seg.columns if c not in ("segment", "n")]
    lines.append("| segment | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for _, r in seg.iterrows():
        lines.append(f"| {r['segment']} | {r['n']} | " + " | ".join(f"{r[c]}%" for c in cols) + " |")
    lines += ["", '<p align="center">',
              '  <img src="docs/figures/comparison.png" width="85%"><br>',
              '  <img src="docs/figures/error_profile.png" width="85%"><br>',
              '  <img src="docs/figures/price_bands.png" width="85%"><br>',
              '  <img src="docs/figures/coverage_width.png" width="46%">',
              '  <img src="docs/figures/cost_accuracy.png" width="44%"><br>',
              '  <img src="docs/figures/error_ecdf.png" width="46%">',
              '  <img src="docs/figures/calibration.png" width="40%"><br>',
              '  <img src="docs/figures/segments.png" width="85%">',
              "</p>", ""]

    block = "\n".join(lines)
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
                    f"<!-- RESULTS:BEGIN -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.DOTALL)
    (ROOT / "README.md").write_text(readme)

    # committed figure copies live in docs/figures (reports/ is git-ignored)
    fig_src = ROOT / "reports/figures"
    fig_dst = ROOT / "docs/figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for f in fig_src.glob("*.png"):
        fig_dst.joinpath(f.name).write_bytes(f.read_bytes())
    print("README results updated; figures copied to docs/figures")


if __name__ == "__main__":
    main()
