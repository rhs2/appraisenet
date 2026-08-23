"""Environment handling: .env loading and typed lookups.

Locations of private resources (the listings database, tracking servers) are never
hard-coded; they come from the environment so the repository carries no secrets.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency on python-dotenv at import time)."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def env(key: str, default: str | None = None) -> str | None:
    load_dotenv()
    return os.environ.get(key, default)


def db_path() -> Path | None:
    """Path to the private listings DB, or None when it is not available."""
    v = env("APPRAISENET_DB", "data/listings.db")
    p = Path(v) if Path(v).is_absolute() else ROOT / v
    return p if p.exists() else None
