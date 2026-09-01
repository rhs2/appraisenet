"""Generate every table and numeric macro in the paper from the persisted study
artifacts (reports/results.csv, comparison_stats.csv, pairwise_mape.csv,
error_profile.csv, price_bands.csv, segments.csv, run_meta.json for the corpus-scale
study; reports/pilot/* for the pilot tier). No number in the manuscript is typed by
hand: edit the study, re-run the benchmark, re-run this script, rebuild the paper."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
TABLES = PAPER / "tables"
TABLES.mkdir(exist_ok=True)

DISPLAY = {
    "anchored_blend": "Anchored blend (two engines)",
    "anchored_hybrid": "Anchored hybrid (residual)",
    "anchored_lgbm": "Anchored LightGBM",
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
    "anchored_blend": "Anch.\\ blend", "anchored_hybrid": "Anch.\\ hybrid",
    "anchored_lgbm": "Anch.\\ LGBM", "hybrid_lgbm_text": "Text hybrid",
    "stack": "Stack", "blend_lgbm_catboost": "Blend",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "embed_mlp": "MLP", "ft_transformer": "FT-Trans.", "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest", "knn_comparables": "$k$-NN",
    "ridge": "Ridge", "elasticnet": "Elastic net",
}


def _fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def _fmt_minutes(seconds: float) -> str:
    """Fit time in minutes, with a decimal while the models are still quick: at pilot
    scale most of the zoo finishes in under a minute and would otherwise all read 0."""
    minutes = seconds / 60
    return f"{minutes:.0f}" if minutes >= 10 else f"{minutes:.1f}"


def _results_table(res: pd.DataFrame, stats, label: str, caption: str, star: bool) -> str:
    rows = []
    for _, r in res.iterrows():
        ci = ""
        if stats is not None and r["model"] in stats.index:
            s = stats.loc[r["model"]]
            ci = f"[{s['mape_ci95_low']:.2f}, {s['mape_ci95_high']:.2f}]"
        rows.append(
            f"{DISPLAY.get(r['model'], r['model'])} & {r['family']} & {r['cv_mape_pct']:.2f} & "
            f"\\textbf{{{r['holdout_mape_pct']:.2f}}} & {ci} & {r['holdout_median_ape_pct']:.2f} & "
            f"{r['holdout_within_10_pct']:.1f} & {r['holdout_coverage_pct']:.1f} & "
            f"{r['holdout_width_pct_of_price']:.1f} & {_fmt_minutes(r['fit_seconds'])} \\\\")
    env = "table*" if star else "table"
    return ("\\begin{" + env + "}[t]\n\\centering\n"
            "\\caption{" + caption + "}\n"
            "\\label{" + label + "}\n\\footnotesize\\setlength{\\tabcolsep}{5pt}\n"
            "\\begin{tabular}{llrrcrrrrr}\n\\toprule\n"
            "Model & Family & CV & Holdout & 95\\,\\% CI & Median & $\\le$10\\,\\% & Coverage & Width & Fit (min) \\\\\n"
            "\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{" + env + "}\n")


def _pair_macros(cmd, root: Path) -> list[str]:
    """Macros for the two claims the paper makes about specific pairs of models, each
    with its own paired-bootstrap interval from reports/pairwise_mape.csv."""
    path = root / "reports/pairwise_mape.csv"
    if not path.exists():
        return [cmd("TiedPairs", "--"), cmd("TotalPairs", "--")]
    pw = pd.read_csv(path)
    out = [cmd("TiedPairs", str(int((~pw["separated"]).sum()))),
           cmd("TotalPairs", str(len(pw)))]

    def gap(name: str, a: str, b: str) -> list[str]:
        """b minus a, i.e. what a buys over b, positive when a is better."""
        row = pw[(pw.model_a == a) & (pw.model_b == b)]
        flip = row.empty
        if flip:
            row = pw[(pw.model_a == b) & (pw.model_b == a)]
        if row.empty:
            return [cmd(name, "--"), cmd(name + "CILow", "--"), cmd(name + "CIHigh", "--")]
        r = row.iloc[0]
        sign = 1.0 if flip else -1.0
        lo, hi = sorted((sign * r["delta_ci95_low"], sign * r["delta_ci95_high"]))
        return [cmd(name, f"{sign * r['delta_mape_pct']:.3f}"),
                cmd(name + "CILow", f"{lo:.3f}"), cmd(name + "CIHigh", f"{hi:.3f}")]

    out += gap("TextGain", "hybrid_lgbm_text", "lightgbm")
    out += gap("AnchorGain", "anchored_lgbm", "lightgbm")
    return out


def _profile_macros(cmd, prof, stats, res, bands) -> list[str]:
    """Macros for the mean-versus-median finding: who wins each summary, by how much,
    and what the tail behind the mean looks like."""
    if prof is None:
        return []
    P = prof.set_index("model")
    champ = res.iloc[0]["model"]
    best_med = prof.sort_values("p50").iloc[0]["model"]
    out = [cmd("BestMedianName", DISPLAY.get(best_med, best_med)),
           cmd("BestMedian", f"{P.loc[best_med, 'p50']:.2f}"),
           cmd("BestMedianMape", f"{P.loc[best_med, 'mape_pct']:.2f}"),
           cmd("ChampionWithinFive", f"{P.loc[champ, 'within_5_pct']:.1f}"),
           cmd("BestMedianWithinFive", f"{P.loc[best_med, 'within_5_pct']:.1f}"),
           cmd("ChampionPNN", f"{P.loc[champ, 'p99']:.0f}"),
           cmd("BestMedianPNN", f"{P.loc[best_med, 'p99']:.0f}"),
           cmd("ChampionOverFifty", f"{P.loc[champ, 'over_50_pct']:.2f}"),
           cmd("BestMedianOverFifty", f"{P.loc[best_med, 'over_50_pct']:.2f}"),
           cmd("ChampionTailShare", f"{P.loc[champ, 'tail_share_of_error_pct']:.0f}"),
           cmd("BestMedianTailShare", f"{P.loc[best_med, 'tail_share_of_error_pct']:.0f}"),
           cmd("ChampionOverHundred", f"{P.loc[champ, 'over_100_pct']:.2f}"),
           cmd("BestMedianOverHundred", f"{P.loc[best_med, 'over_100_pct']:.2f}"),
           cmd("BestMedianOverpriced", f"{P.loc[best_med, 'over_50_overpriced_pct']:.0f}"),
           cmd("ChampionOverpriced", f"{P.loc[champ, 'over_50_overpriced_pct']:.0f}"),
           cmd("BestMedianOverTwentyFive", f"{P.loc[best_med, 'over_25_pct']:.1f}")]
    if "median_delta_vs_best" in stats.columns and champ in stats.index:
        c = stats.loc[champ]
        out += [cmd("ChampionMedianGap", f"{c['median_delta_vs_best']:.3f}"),
                cmd("ChampionMedianGapCILow", f"{c['median_delta_ci95_low']:.3f}"),
                cmd("ChampionMedianGapCIHigh", f"{c['median_delta_ci95_high']:.3f}")]
    if bands is not None:
        w = bands.pivot(index="band", columns="model", values="mape_pct")
        n_by = bands.drop_duplicates("band").set_index("band")["n"]
        for key, label in (("Cheap", "under $10k"), ("Mid", "$20k-40k"), ("Upper", "over $40k")):
            if label in w.index:
                out += [cmd(key + "BandChampion", f"{w.loc[label, champ]:.2f}"),
                        cmd(key + "BandBestMedian", f"{w.loc[label, best_med]:.2f}")]
        if "under $10k" in n_by.index:
            out += [cmd("CheapBandN", _fmt_int(n_by["under $10k"])),
                    cmd("CheapBandShare", f"{n_by['under $10k'] / n_by.sum() * 100:.1f}")]
    return out


def _scaling_macros(cmd, R: pd.DataFrame, P: pd.DataFrame) -> list[str]:
    """How much every configuration that ran at both scales gained from the larger corpus."""
    both = [m for m in R.index if m in P.index]
    if not both:
        return [cmd("ScaleGainMin", "--"), cmd("ScaleGainMax", "--"), cmd("ScaleGainModels", "0")]
    gain = P.loc[both, "holdout_mape_pct"] - R.loc[both, "holdout_mape_pct"]
    return [cmd("ScaleGainMin", f"{gain.min():.1f}"), cmd("ScaleGainMax", f"{gain.max():.1f}"),
            cmd("ScaleGainModels", str(len(both)))]


def main() -> None:
    res = pd.read_csv(ROOT / "reports/results.csv")
    stats = pd.read_csv(ROOT / "reports/comparison_stats.csv").set_index("model")
    seg = pd.read_csv(ROOT / "reports/segments.csv")
    meta = json.loads((ROOT / "reports/run_meta.json").read_text())

    n_models = len(res)
    caption = (f"The corpus-scale study: all {n_models} scalable configurations under the "
               "AppraiseNet Evaluation Protocol on " + _fmt_int(meta["rows"]) + " listings. "
               "CV is the 5-fold out-of-fold MAPE on the training partition; every holdout "
               "column is a single scoring pass on the untouched 10\\,\\% split. The "
               "95\\,\\% CI is a paired bootstrap (4{,}000 resamples of the shared holdout). "
               "Coverage and width describe the split-conformal 80\\,\\% interval. "
               "All values are percentages.")
    (TABLES / "results.tex").write_text(_results_table(res, stats, "tab:results", caption, star=True))

    # ---- pilot tier table --------------------------------------------------
    pres = pd.read_csv(ROOT / "reports/pilot/results.csv")
    pstats_path = ROOT / "reports/pilot/comparison_stats.csv"
    pstats = pd.read_csv(pstats_path).set_index("model") if pstats_path.exists() else None
    pmeta = json.loads((ROOT / "reports/pilot/run_meta.json").read_text())
    pcaption = (f"The pilot tier: the full {len(pres)}-model zoo on the "
                + _fmt_int(pmeta["rows"]) + "-listing corpus snapshot of August 2026, "
                "same protocol, same columns as Table~\\ref{tab:results}.")
    (TABLES / "pilot.tex").write_text(_results_table(pres, pstats, "tab:pilot", pcaption, star=True))

    # ---- segment table -----------------------------------------------------
    models = [c for c in seg.columns if c not in ("segment", "n")]
    head = " & ".join(SHORT.get(m, m) for m in models)
    seg_rows = []
    for _, r in seg.iterrows():
        seg_name = str(r["segment"]).replace("$", "\\$")
        vals = " & ".join(f"{r[m]:.1f}" for m in models)
        seg_rows.append(f"{seg_name} & {int(r['n']):,} & {vals} \\\\")
    (TABLES / "segments.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Holdout MAPE (\\%) by segment for the " + str(len(models)) +
        " strongest corpus-scale models. The hard segments are shared: cheap, old, "
        "high-mileage and private-party cars.}\n"
        "\\label{tab:segments}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lr" + "r" * len(models) + "}\n\\toprule\n"
        "Segment & $n$ & " + head + " \\\\\n\\midrule\n" + "\n".join(seg_rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---- error-profile table (the two summaries and the tail) --------------
    prof_path = ROOT / "reports/error_profile.csv"
    prof = pd.read_csv(prof_path) if prof_path.exists() else None
    if prof is not None:
        st = stats.reindex(prof["model"])
        heavy = prof.sort_values("tail_share_of_error_pct").iloc[-1]
        prows = []
        for (_, r), (_, s_) in zip(prof.iterrows(), st.iterrows()):
            ci = (f"[{s_['median_ci95_low']:.2f}, {s_['median_ci95_high']:.2f}]"
                  if "median_ci95_low" in st.columns and pd.notna(s_.get("median_ci95_low")) else "")
            prows.append(f"{DISPLAY.get(r['model'], r['model'])} & {r['p50']:.2f} & {ci} & "
                         f"{r['mape_pct']:.2f} & {r['within_5_pct']:.1f} & {r['p90']:.1f} & "
                         f"{r['p99']:.1f} & {r['over_50_pct']:.2f} & "
                         f"{r['tail_share_of_error_pct']:.1f} \\\\")
        (TABLES / "profile.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{The error distribution behind the leaderboard, ordered by MAPE. "
            "The median describes the typical car and the mean describes the tail; the "
            "anchored residual engines hold the best medians and the worst means, because "
            + f"{heavy['over_25_pct']:.0f}" + "\\,\\% of their errors carry "
            + f"{heavy['tail_share_of_error_pct']:.0f}" + "\\,\\% of their total error. "
            "The median CI is the "
            "same paired bootstrap as Table~\\ref{tab:results}. All values are percentages "
            "except the last column, which is the share of summed absolute percentage error "
            "contributed by cars missed by more than 25\\,\\%.}\n"
            "\\label{tab:profile}\n\\footnotesize\\setlength{\\tabcolsep}{5pt}\n"
            "\\begin{tabular}{lrcrrrrrr}\n\\toprule\n"
            "Model & Median & 95\\,\\% CI & MAPE & $\\le$5\\,\\% & p90 & p99 & $>$50\\,\\% & Tail share \\\\\n"
            "\\midrule\n" + "\n".join(prows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---- price-band table (where the two summaries disagree) ---------------
    band_path = ROOT / "reports/price_bands.csv"
    bands = pd.read_csv(band_path) if band_path.exists() else None
    if bands is not None:
        want = [res.iloc[0]["model"], "anchored_lgbm", "anchored_blend", "anchored_hybrid",
                "ft_transformer"]
        cols = [m for m in dict.fromkeys(want) if m in set(bands["model"])]
        wide = bands.pivot(index="band", columns="model", values="mape_pct")
        n_by = bands.drop_duplicates("band").set_index("band")["n"]
        order = [b for b in ("under $10k", "$10k-20k", "$20k-40k", "over $40k") if b in wide.index]
        brows = [str(b).replace("$", "\\$") + f" & {int(n_by[b]):,} & "
                 + " & ".join(f"{wide.loc[b, m]:.2f}" for m in cols) + " \\\\"
                 for b in order]
        (TABLES / "bands.tex").write_text(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Holdout MAPE (\\%) by price band. The anchored residual engines are the "
            "most accurate models in the two upper bands and the least accurate by far in the "
            "cheapest one, which holds only " + (f"{n_by[order[0]] / n_by.sum() * 100:.1f}" if order else "--") +
            "\\,\\% of the holdout yet decides the corpus mean.}\n"
            "\\label{tab:bands}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lr" + "r" * len(cols) + "}\n\\toprule\n"
            "Band & $n$ & " + " & ".join(SHORT.get(m, m) for m in cols) + " \\\\\n\\midrule\n"
            + "\n".join(brows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---- numeric macros ----------------------------------------------------
    R = res.set_index("model")
    P = pres.set_index("model")

    def cmd(name: str, value: str) -> str:
        return "\\newcommand{\\" + name + "}{" + value + "}"

    def mape(frame, model):
        return f"{frame.loc[model, 'holdout_mape_pct']:.2f}" if model in frame.index else "--"

    champ = res.iloc[0]
    cs = stats.loc[champ["model"]] if champ["model"] in stats.index else None
    cqr = meta.get("cqr", {})
    cov = res["holdout_coverage_pct"]
    pchamp = pres.iloc[0]

    macros = [
        cmd("NRows", _fmt_int(meta["rows"])),
        cmd("NTrain", _fmt_int(meta["train"])),
        cmd("NHoldout", _fmt_int(meta["holdout"])),
        cmd("NModels", str(n_models)),
        cmd("ChampionName", DISPLAY.get(champ["model"], champ["model"])),
        cmd("ChampionMape", f"{champ['holdout_mape_pct']:.2f}"),
        cmd("ChampionCILow", f"{cs['mape_ci95_low']:.2f}" if cs is not None else "--"),
        cmd("ChampionCIHigh", f"{cs['mape_ci95_high']:.2f}" if cs is not None else "--"),
        cmd("ChampionMedian", f"{champ['holdout_median_ape_pct']:.2f}"),
        cmd("ChampionWithin", f"{champ['holdout_within_10_pct']:.1f}"),
        cmd("ChampionWidth", f"{champ['holdout_width_pct_of_price']:.1f}"),
        cmd("LgbmMape", mape(R, "lightgbm")),
        cmd("TextHybridMape", mape(R, "hybrid_lgbm_text")),
        cmd("AnchBlendMape", mape(R, "anchored_blend")),
        cmd("AnchHybridMape", mape(R, "anchored_hybrid")),
        cmd("AnchLgbmMape", mape(R, "anchored_lgbm")),
        cmd("BestBoostMape", f"{res[res.family == 'boosting']['holdout_mape_pct'].min():.2f}"),
        cmd("WorstBoostMape", f"{res[res.family == 'boosting']['holdout_mape_pct'].max():.2f}"),
        cmd("BestDeepMape", f"{res[res.family == 'deep tabular']['holdout_mape_pct'].min():.2f}"
            if (res.family == "deep tabular").any() else "--"),
        cmd("BestAnchoredMape", f"{res[res.family == 'anchored']['holdout_mape_pct'].min():.2f}"
            if (res.family == "anchored").any() else "--"),
        cmd("WorstMape", f"{res['holdout_mape_pct'].max():.2f}"),
        cmd("CovMin", f"{cov.min():.1f}"),
        cmd("CovMax", f"{cov.max():.1f}"),
        cmd("CQRCoverage", f"{cqr.get('cqr_holdout_coverage_pct', float('nan')):.1f}"),
        cmd("CQRWidth", f"{cqr.get('cqr_holdout_width_pct_of_price', float('nan')):.1f}"),
        cmd("CQRWidthGain", f"{champ['holdout_width_pct_of_price'] - cqr['cqr_holdout_width_pct_of_price']:.1f}"
            if "cqr_holdout_width_pct_of_price" in cqr else "--"),
        # ---- compute ----
        cmd("StudyHours", f"{res['fit_seconds'].sum() / 3600:.0f}"),
        cmd("ChampionFitMin", f"{champ['fit_seconds'] / 60:.0f}"),
        cmd("AnchLgbmFitMin", f"{R.loc['anchored_lgbm', 'fit_seconds'] / 60:.0f}"
            if "anchored_lgbm" in R.index else "--"),
        cmd("CostRatio", f"{champ['fit_seconds'] / R.loc['anchored_lgbm', 'fit_seconds']:.0f}"
            if "anchored_lgbm" in R.index else "--"),
        cmd("DeepFitHours", f"{R['fit_seconds'].max() / 3600:.0f}"),
        cmd("SlowestModel", DISPLAY.get(R['fit_seconds'].idxmax(), R['fit_seconds'].idxmax())),
        # ---- what 30x the corpus was worth, model by model ----
        *_scaling_macros(cmd, R, P),
        # ---- pairwise statistics ----
        *_pair_macros(cmd, ROOT),
        # ---- the two summaries and the tail ----
        *_profile_macros(cmd, prof, stats, res, bands),
        # ---- pilot tier ----
        cmd("PilotRows", _fmt_int(pmeta["rows"])),
        cmd("PilotTrain", _fmt_int(pmeta["train"])),
        cmd("PilotHoldout", _fmt_int(pmeta["holdout"])),
        cmd("PilotNModels", str(len(pres))),
        cmd("PilotChampionName", DISPLAY.get(pchamp["model"], pchamp["model"])),
        cmd("PilotChampionMape", f"{pchamp['holdout_mape_pct']:.2f}"),
        cmd("PilotChampionMedian", f"{pchamp['holdout_median_ape_pct']:.2f}"),
        cmd("PilotChampionWithin", f"{pchamp['holdout_within_10_pct']:.1f}"),
        cmd("PilotLgbmMape", mape(P, "lightgbm")),
        cmd("PilotTextGain", f"{P.loc['lightgbm', 'holdout_mape_pct'] - P.loc['hybrid_lgbm_text', 'holdout_mape_pct']:.2f}"
            if {"lightgbm", "hybrid_lgbm_text"} <= set(P.index) else "--"),
        cmd("PilotStackMape", mape(P, "stack")),
        cmd("PilotKnnMape", mape(P, "knn_comparables")),
        cmd("PilotRidgeMape", mape(P, "ridge")),
        cmd("PilotWorstMape", f"{pres['holdout_mape_pct'].max():.2f}"),
        cmd("PilotBestDeepMape", f"{pres[pres.family == 'deep tabular']['holdout_mape_pct'].min():.2f}"
            if (pres.family == "deep tabular").any() else "--"),
    ]
    (TABLES / "macros.tex").write_text("\n".join(macros) + "\n")
    print(f"tables: results ({n_models} models), pilot ({len(pres)} models), "
          f"segments ({len(seg)} rows), {len(macros)} macros")


if __name__ == "__main__":
    main()
