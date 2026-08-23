"""YAML experiment configuration -> typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .env import ROOT


@dataclass
class ProtocolCfg:
    seed: int = 42
    holdout_frac: float = 0.10
    folds: int = 5
    price_min: int = 2_000
    price_max: int = 100_000
    year_min: int = 1990
    target_coverage: float = 0.80
    current_year: int = 2026


@dataclass
class Config:
    name: str = "default"
    models: list[str] = field(default_factory=list)   # empty = the full zoo
    protocol: ProtocolCfg = field(default_factory=ProtocolCfg)
    max_rows: int | None = None                        # subsample for smoke runs
    text_max_features: int = 20_000
    reports_dir: str = "reports"
    torch_epochs: int = 40
    torch_batch: int = 1024

    @property
    def out_dir(self) -> Path:
        d = ROOT / self.reports_dir
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(overrides or {})
    proto = ProtocolCfg(**raw.pop("protocol", {}))
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    cfg.protocol = proto
    return cfg
