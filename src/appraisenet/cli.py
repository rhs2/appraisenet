"""Command line interface.

    appraisenet benchmark --config configs/default.yaml
    appraisenet report                 # rebuild figures from saved artifacts
    appraisenet data check             # dataset (or synthetic) sanity summary
    appraisenet serve --port 8080      # the prediction API
    appraisenet tracking replay        # push reports/runs.jsonl into MLflow
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="appraisenet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run the model zoo under the protocol")
    b.add_argument("--config", default="configs/default.yaml")
    b.add_argument("--models", nargs="*", default=None)
    b.add_argument("--max-rows", type=int, default=None)

    sub.add_parser("report", help="rebuild figures from saved artifacts")

    d = sub.add_parser("data", help="dataset utilities")
    d.add_argument("action", choices=["check", "ingest"])
    d.add_argument("--source", default=None,
                   help="ingest: new listings to append - a sqlite .db (table `listings`) or a .csv")

    s = sub.add_parser("serve", help="run the prediction API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)

    t = sub.add_parser("tracking", help="tracking utilities")
    t.add_argument("action", choices=["replay"])

    tp = sub.add_parser("train-production", help="fit + stage the production model (promote-or-rollback)")
    tp.add_argument("--config", default="configs/default.yaml")
    tp.add_argument("--force", action="store_true")

    mo = sub.add_parser("monitor", help="drift report over recent prediction traffic")
    mo.add_argument("--hours", type=int, default=168)

    a = ap.parse_args(argv)
    if a.cmd == "benchmark":
        from .benchmark import run_benchmark
        from .config import load_config
        over = {}
        if a.models:
            over["models"] = a.models
        if a.max_rows:
            over["max_rows"] = a.max_rows
        run_benchmark(load_config(a.config, over))
        return 0
    if a.cmd == "report":
        from .compare import comparison_stats
        from .config import load_config
        from .evaluate import make_figures
        out = load_config(None).out_dir
        stats = comparison_stats(out)
        if stats is not None:
            tied = stats.loc[stats["tied_with_champion"], "model"].tolist()
            print("paired bootstrap, statistically tied for best:", ", ".join(tied))
        for p in make_figures(out):
            print("figure:", p)
        return 0
    if a.cmd == "data":
        from . import db
        from .config import load_config
        if a.action == "ingest":
            import json as _json
            if not a.source:
                ap.error("data ingest requires --source <file.db|file.csv>")
            print(_json.dumps(db.ingest(a.source, load_config(None).protocol), indent=1))
            return 0
        from .data import engineer, load_listings
        df, synthetic = load_listings()
        df = engineer(df, load_config(None).protocol)
        print("storage:", db.describe())
        print(f"{'SYNTHETIC' if synthetic else 'private'} corpus: {len(df):,} listings after protocol filters")
        print(df[["price", "year", "mileage"]].describe().round(0).to_string())
        print("makes:", df["make"].nunique(), "| models:", df["model"].nunique(),
              "| with text:", int((df['description'].str.len() > 40).sum()))
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("appraisenet.serve:app", host=a.host, port=a.port)
        return 0
    if a.cmd == "train-production":
        import json as _json

        from .config import load_config
        from .registry import train_production
        print(_json.dumps(train_production(load_config(a.config), force=a.force), indent=1))
        return 0
    if a.cmd == "monitor":
        import json as _json

        from .config import load_config
        from .monitor import report
        print(_json.dumps(report(load_config(None), hours=a.hours), indent=1))
        return 0
    if a.cmd == "tracking":
        from .config import load_config
        from .tracking import replay_jsonl
        replay_jsonl(load_config(None).out_dir)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
