"""Generate every table and numeric macro in the paper from the persisted study
artifacts (reports/results.csv, comparison_stats.csv, segments.csv, run_meta.json).
No number in the manuscript is typed by hand: edit the study, re-run the benchmark,
re-run this script, rebuild the paper."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
TABLES = PAPER / "tables"
TABLES.mkdir(exist_ok=True)

DISPLAY = {
    "hybrid_lgbm_text": "LightGBM + text residual",
    "stack": "Stacked ensemble",
    "blend_lgbm_catboost": "LightGBM+CatBoost blend",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "embed_mlp": "Embedding MLP",
    "ft_transformer": "FT-Transformer",
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
    "knn_comparables": "$k$-NN comparables",
    "ridge": "Ridge",
    "elasticnet": "Elastic net",
}
SHORT = {
    "hybrid_lgbm_text": "Text hybrid", "stack": "Stack", "blend_lgbm_catboost": "Blend",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
}


def main() -> None:
    res = pd.read_csv(ROOT / "reports/results.csv")
    stats = pd.read_csv(ROOT / "reports/comparison_stats.csv").set_index("model")
    seg = pd.read_csv(ROOT / "reports/segments.csv")
    meta = json.loads((ROOT / "reports/run_meta.json").read_text())

    # ---- main results table -----------------------------------------------
    rows = []
    for _, r in res.iterrows():
        s = stats.loc[r["model"]]
        ci = f"[{s['mape_ci95_low']:.2f}, {s['mape_ci95_high']:.2f}]"
        rows.append(
            f"{DISPLAY[r['model']]} & {r['family']} & {r['cv_mape_pct']:.2f} & "
            f"\\textbf{{{r['holdout_mape_pct']:.2f}}} & {ci} & {r['holdout_median_ape_pct']:.2f} & "
            f"{r['holdout_within_10_pct']:.1f} & {r['holdout_coverage_pct']:.1f} & "
            f"{r['holdout_width_pct_of_price']:.1f} \\\\")
    (TABLES / "results.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{All thirteen models under the AppraiseNet Evaluation Protocol. CV is the "
        "5-fold out-of-fold MAPE on the training partition; every holdout column is a single "
        "scoring pass on the untouched 10\\,\\% split. The 95\\,\\% CI is a paired bootstrap "
        "(4{,}000 resamples of the shared holdout). Coverage and width describe the "
        "split-conformal 80\\,\\% interval. All values are percentages.}\n"
        "\\label{tab:results}\n\\footnotesize\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{llrrcrrrr}\n\\toprule\n"
        "Model & Family & CV & Holdout & 95\\,\\% CI & Median & $\\le$10\\,\\% & Coverage & Width \\\\\n"
        "\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---- segment table -----------------------------------------------------
    models = [c for c in seg.columns if c not in ("segment", "n")]
    head = " & ".join(SHORT.get(m, m) for m in models)
    seg_rows = []
    for _, r in seg.iterrows():
        seg_name = str(r["segment"]).replace("$", "\\$")
        vals = " & ".join(f"{r[m]:.1f}" for m in models)
        seg_rows.append(f"{seg_name} & {r['n']:,} & {vals} \\\\")
    (TABLES / "segments.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Holdout MAPE (\\%) by segment for the five strongest models. "
        "The hard segments are shared: cheap, old, high-mileage and private-party cars.}\n"
        "\\label{tab:segments}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lr" + "r" * len(models) + "}\n\\toprule\n"
        f"Segment & $n$ & {head} \\\\\n\\midrule\n" + "\n".join(seg_rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---- numeric macros ----------------------------------------------------
    champ = res.iloc[0]
    cs = stats.loc[champ["model"]]
    lgbm = res.set_index("model").loc["lightgbm"]
    stack_s = stats.loc["stack"]
    cqr = meta.get("cqr", {})
    cov = res["holdout_coverage_pct"]

    def cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    macros = [
        cmd("NRows", f"{meta['rows']:,}".replace(",", "{,}")),
        cmd("NTrain", f"{meta['train']:,}".replace(",", "{,}")),
        cmd("NHoldout", f"{meta['holdout']:,}".replace(",", "{,}")),
        cmd("ChampionName", DISPLAY[champ["model"]]),
        cmd("ChampionMape", f"{champ['holdout_mape_pct']:.2f}"),
        cmd("ChampionCILow", f"{cs['mape_ci95_low']:.2f}"),
        cmd("ChampionCIHigh", f"{cs['mape_ci95_high']:.2f}"),
        cmd("ChampionMedian", f"{champ['holdout_median_ape_pct']:.2f}"),
        cmd("ChampionWithin", f"{champ['holdout_within_10_pct']:.1f}"),
        cmd("LgbmMape", f"{lgbm['holdout_mape_pct']:.2f}"),
        cmd("TextGain", f"{lgbm['holdout_mape_pct'] - champ['holdout_mape_pct']:.2f}"),
        cmd("StackMape", f"{res.set_index('model').loc['stack', 'holdout_mape_pct']:.2f}"),
        cmd("StackDeltaLow", f"{stack_s['delta_ci95_low']:.2f}"),
        cmd("StackDeltaHigh", f"{stack_s['delta_ci95_high']:.2f}"),
        cmd("BestBoostMape", f"{res[res.family == 'boosting']['holdout_mape_pct'].min():.2f}"),
        cmd("BestDeepMape", f"{res[res.family == 'deep tabular']['holdout_mape_pct'].min():.2f}"),
        cmd("WorstMape", f"{res['holdout_mape_pct'].max():.2f}"),
        cmd("CovMin", f"{cov.min():.1f}"),
        cmd("CovMax", f"{cov.max():.1f}"),
        cmd("CQRCoverage", f"{cqr.get('cqr_holdout_coverage_pct', float('nan')):.1f}"),
        cmd("CQRWidth", f"{cqr.get('cqr_holdout_width_pct_of_price', float('nan')):.1f}"),
        cmd("ChampionWidth", f"{champ['holdout_width_pct_of_price']:.1f}"),
    ]
    (TABLES / "macros.tex").write_text("\n".join(macros) + "\n")
    print(f"tables: results ({len(res)} models), segments ({len(seg)} rows), {len(macros)} macros")


if __name__ == "__main__":
    main()
