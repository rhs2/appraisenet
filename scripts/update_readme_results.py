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

    top = res.iloc[0]
    lines = ["## Results", ""]
    corpus = "synthetic corpus (illustrative only)" if meta["synthetic"] else \
        f"{meta['rows']:,} real listings ({meta['train']:,} train / {meta['holdout']:,} holdout)"
    lines.append(f"**{corpus}**, protocol as below; champion: **{top['model']}** at "
                 f"**{top['holdout_mape_pct']}% holdout MAPE** "
                 f"(median {top['holdout_median_ape_pct']}%, {top['holdout_within_10_pct']}% of cars within 10%).")
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
        lines += ["", "† statistically tied with the champion: the 95% paired-bootstrap interval of the "
                      "MAPE gap (4,000 resamples of the shared holdout) includes zero."]
    if "cqr" in meta:
        c = meta["cqr"]
        lines += ["", f"Production interval (CQR on the champion): **{c['cqr_holdout_coverage_pct']}% coverage** "
                      f"at a median width of {c['cqr_holdout_width_pct_of_price']}% of price."]
    lines += ["", "**Holdout MAPE by segment (top models):**", ""]
    cols = [c for c in seg.columns if c not in ("segment", "n")]
    lines.append("| segment | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for _, r in seg.iterrows():
        lines.append(f"| {r['segment']} | {r['n']} | " + " | ".join(f"{r[c]}%" for c in cols) + " |")
    lines += ["", '<p align="center">',
              '  <img src="docs/figures/comparison.png" width="85%"><br>',
              '  <img src="docs/figures/coverage_width.png" width="46%">',
              '  <img src="docs/figures/error_ecdf.png" width="44%"><br>',
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
