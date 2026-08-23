"""Experiment tracking with MLflow, degrading gracefully to a local JSONL log.

Every benchmark cell becomes one MLflow run under an experiment named after the
config. Params, metrics, the config file, confusion matrices and the exported
best model are logged. The default store is `sqlite:///reports/mlflow.db`
(MLflow's file store is in maintenance mode); point `tracking.uri` or
`MLFLOW_TRACKING_URI` at a server to log remotely.

Whatever the backend, every run is also appended to `reports/runs.jsonl`.
If MLflow is missing or unreachable the JSONL file is the record, and
`replay_jsonl` can push it into MLflow later (`appraisenet tracking replay`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def default_uri(reports_dir: Path) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(Path(reports_dir) / 'mlflow.db').resolve()}"


class Tracker:
    def __init__(self, experiment: str, reports_dir: Path, enabled: bool = True, tracking_uri: str | None = None):
        self.experiment = experiment
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.reports_dir / "runs.jsonl"
        self.mlflow = None
        self.uri = tracking_uri or default_uri(self.reports_dir)
        if enabled:
            try:
                import mlflow

                logging.getLogger("mlflow").setLevel(logging.WARNING)
                logging.getLogger("alembic").setLevel(logging.WARNING)
                if self.uri.startswith("file:") or self.uri.startswith("./") or self.uri.startswith("/"):
                    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                mlflow.set_tracking_uri(self.uri)
                mlflow.set_experiment(experiment)
                self.mlflow = mlflow
            except Exception as exc:  # pragma: no cover
                print(f"[tracking] MLflow unavailable ({exc}); logging to {self.jsonl}")

    @property
    def backend(self) -> str:
        return f"mlflow ({self.uri})" if self.mlflow else "jsonl"

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None):
        record: dict[str, Any] = {
            "run": name,
            "experiment": self.experiment,
            "tags": tags or {},
            "params": {},
            "metrics": {},
            "artifacts": [],
            "started": time.time(),
        }
        if self.mlflow:
            with self.mlflow.start_run(run_name=name, tags=tags):
                yield _Run(self, record)
        else:
            yield _Run(self, record)
        record["finished"] = time.time()
        with open(self.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


class _Run:
    def __init__(self, tracker: Tracker, record: dict[str, Any]):
        self.t = tracker
        self.record = record

    def params(self, params: dict[str, Any]) -> None:
        flat = {k: _short(v) for k, v in params.items()}
        self.record["params"].update(flat)
        if self.t.mlflow:
            self.t.mlflow.log_params(flat)

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self.record["metrics"].update(clean)
        if self.t.mlflow:
            self.t.mlflow.log_metrics(clean, step=step)

    def artifact(self, path: Path | str) -> None:
        self.record["artifacts"].append(str(path))
        if self.t.mlflow and Path(path).exists():
            self.t.mlflow.log_artifact(str(path))

    def model(self, estimator, name: str = "model") -> None:
        """Log the fitted pipeline: MLflow sklearn flavour when its serializer is
        available, otherwise the joblib file as a plain artifact."""
        if not self.t.mlflow:
            return
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(estimator, name=name)
            return
        except Exception as exc:
            print(f"[tracking] mlflow.sklearn flavour unavailable ({exc}); logging joblib artifact")
        import tempfile

        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.joblib"
            joblib.dump(estimator, path)
            self.t.mlflow.log_artifact(str(path), artifact_path=name)
            self.record["artifacts"].append(f"{name}/{name}.joblib")


def replay_jsonl(jsonl: Path, tracking_uri: str | None = None, experiment: str | None = None) -> int:
    """Push runs recorded in runs.jsonl into an MLflow store. Returns the count."""
    import mlflow

    jsonl = Path(jsonl)
    uri = tracking_uri or default_uri(jsonl.parent)
    if uri.startswith("file:") or uri.startswith("./") or uri.startswith("/"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    mlflow.set_tracking_uri(uri)
    n = 0
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            mlflow.set_experiment(experiment or rec.get("experiment") or "replayed")
            tags = dict(rec.get("tags") or {})
            tags["replayed_from"] = jsonl.name
            with mlflow.start_run(run_name=rec["run"], tags=tags):
                if rec.get("params"):
                    mlflow.log_params({k: _short(v) for k, v in rec["params"].items()})
                if rec.get("metrics"):
                    mlflow.log_metrics({k: float(v) for k, v in rec["metrics"].items()})
                for art in rec.get("artifacts") or []:
                    if Path(art).exists():
                        mlflow.log_artifact(art)
            n += 1
    return n


def _short(v: Any, limit: int = 250) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= limit else s[: limit - 3] + "..."
