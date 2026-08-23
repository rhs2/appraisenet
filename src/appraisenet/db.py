"""Storage layer: one URL, two backends.

`APPRAISENET_DB` accepts either a filesystem path to a SQLite file (the zero-setup
default) or a PostgreSQL DSN such as `postgresql://user:pass@host:5432/appraisenet`
(the production feed). `DATABASE_URL` is honoured as a fallback name so the AWS
deployment, which injects the RDS DSN under that key, needs no extra wiring. Every
reader and writer in the package goes through this module, so switching backends is
a one-line `.env` change and nothing else moves.

Growing the dataset day by day is `appraisenet data ingest --source <file.db|file.csv>`:
rows pass the same quality gates as training, are fingerprinted on the identifying
fields, and only listings never seen before are appended. Re-running it on the same
file inserts nothing, so a daily cron can never duplicate the corpus. Migrating from
SQLite to Postgres is one ingest with the old file as the source and the DSN as the
destination.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .env import ROOT, env

_PG_PREFIXES = ("postgres://", "postgresql://", "postgresql+")
_KEY_COLS = ["year", "make", "model", "trim", "mileage", "price", "seller_type", "region_zip3"]
_INT_KEYS = {"year", "mileage", "price", "region_zip3"}
CANONICAL = ["id", "price", "year", "make", "model", "trim", "mileage", "seller_type",
             "condition", "body_style", "fuel_type", "transmission", "drivetrain",
             "cylinders", "doors", "displacement_l", "engine_hp", "gvwr_class", "series",
             "electrification", "adaptive_cruise", "plant_country", "original_price",
             "region_state", "region_zip3", "description"]


def db_url() -> str:
    return env("APPRAISENET_DB") or env("DATABASE_URL") or "data/listings.db"


def is_postgres(url: str | None = None) -> bool:
    return (url or db_url()).startswith(_PG_PREFIXES)


def sqlite_path(url: str | None = None) -> Path:
    u = url or db_url()
    return Path(u) if Path(u).is_absolute() else ROOT / u


def describe(url: str | None = None) -> str:
    """Human-readable backend description; never includes credentials."""
    u = url or db_url()
    if is_postgres(u):
        return f"postgres @ {u.rsplit('@', 1)[-1]}"
    p = sqlite_path(u)
    return f"sqlite {p}" + ("" if p.exists() else " (absent)")


def _engine(url: str):
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("a PostgreSQL DSN is configured but the driver is missing; "
                           "install it with: pip install 'appraisenet[postgres]'") from exc
    if url.startswith("postgres://"):  # heroku-style scheme sqlalchemy no longer accepts
        url = "postgresql://" + url.removeprefix("postgres://")
    return create_engine(url, pool_pre_ping=True)


def read_listings(url: str | None = None) -> pd.DataFrame | None:
    """The full listings table, or None when the source does not exist yet."""
    u = url or db_url()
    if is_postgres(u):
        from sqlalchemy import inspect, text
        eng = _engine(u)
        if not inspect(eng).has_table("listings"):
            return None
        with eng.connect() as con:
            return pd.read_sql_query(text("select * from listings"), con)
    p = sqlite_path(u)
    return pd.read_sql_query("select * from listings", sqlite3.connect(p)) if p.exists() else None


# ------------------------------------------------------------------ prediction log
def log_prediction(row: dict, out: dict, sqlite_fallback: Path) -> None:
    vals = (datetime.now(timezone.utc).isoformat(timespec="seconds"), out["model_version"],
            json.dumps(row, default=str), out["price"], out["low"], out["high"])
    if is_postgres():
        from sqlalchemy import text
        with _engine(db_url()).begin() as con:
            con.execute(text("""create table if not exists predictions
                (at text, model_version text, request text,
                 price double precision, low double precision, high double precision)"""))
            con.execute(text("insert into predictions values (:a,:b,:c,:d,:e,:f)"),
                        dict(zip("abcdef", vals)))
        return
    sqlite_fallback.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(sqlite_fallback)
    con.execute("""create table if not exists predictions
                   (at text, model_version text, request json, price real, low real, high real)""")
    con.execute("insert into predictions values (?,?,?,?,?,?)", vals)
    con.commit()
    con.close()


def recent_predictions(hours: int, sqlite_fallback: Path) -> list[tuple[str, float]]:
    """(request_json, predicted_price) pairs inside the window, newest backend wins."""
    cut = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    if is_postgres():
        from sqlalchemy import inspect, text
        eng = _engine(db_url())
        if not inspect(eng).has_table("predictions"):
            return []
        with eng.connect() as con:
            return [tuple(r) for r in
                    con.execute(text("select request, price from predictions where at >= :cut"),
                                {"cut": cut})]
    if not sqlite_fallback.exists():
        return []
    con = sqlite3.connect(sqlite_fallback)
    rows = con.execute("select request, price from predictions where at >= ?", (cut,)).fetchall()
    con.close()
    return rows


# ------------------------------------------------------------------ daily ingest
def fingerprint_frame(df: pd.DataFrame) -> pd.Series:
    """Stable identity hash over the identifying fields, robust to int/float/case noise."""
    parts = []
    for c in _KEY_COLS:
        s = df[c] if c in df.columns else pd.Series(pd.NA, index=df.index)
        if c in _INT_KEYS:
            parts.append(pd.to_numeric(s, errors="coerce").round().astype("Int64")
                         .astype("string").fillna(""))
        else:
            parts.append(s.astype("string").str.strip().str.lower().fillna(""))
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + "|" + p
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def _read_source(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    con = sqlite3.connect(path)
    try:
        return pd.read_sql_query("select * from listings", con)
    finally:
        con.close()


def _gate(df: pd.DataFrame, proto) -> pd.DataFrame:
    """The same quality gates the training protocol applies, enforced at the door."""
    df = df.copy()
    for c in ("price", "year", "mileage"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    ok = (df["price"].between(proto.price_min, proto.price_max)
          & df["mileage"].notna() & (df["year"] >= proto.year_min)
          & df.get("make", pd.Series(pd.NA, index=df.index)).notna()
          & df.get("model", pd.Series(pd.NA, index=df.index)).notna())
    return df[ok]


def ingest(source: Path | str, proto, url: str | None = None) -> dict:
    """Append only never-seen listings from a sqlite db or csv; safe to re-run daily."""
    u = url or db_url()
    src = _read_source(Path(source))
    n_source = len(src)
    src = _gate(src, proto)
    n_passed = len(src)

    src = src.loc[~fingerprint_frame(src).duplicated()]
    dest = read_listings(u)
    seen = set(fingerprint_frame(dest)) if dest is not None and len(dest) else set()
    fresh = src.loc[~fingerprint_frame(src).isin(seen)].copy()

    columns = list(dest.columns) if dest is not None else CANONICAL
    for c in columns:
        if c not in fresh.columns:
            fresh[c] = pd.NA
    fresh = fresh[columns]
    start = int(pd.to_numeric(dest["id"], errors="coerce").max()) + 1 \
        if dest is not None and "id" in dest.columns and len(dest) else 1
    if "id" in fresh.columns:
        fresh["id"] = range(start, start + len(fresh))

    summary = {"backend": describe(u), "source_rows": n_source, "passed_gates": n_passed,
               "duplicates": n_passed - len(fresh), "inserted": len(fresh),
               "total_listings": (len(dest) if dest is not None else 0) + len(fresh)}
    if len(fresh):
        _append(fresh, "listings", u)
    log = pd.DataFrame([{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "source": Path(source).name, **{k: summary[k] for k in
                         ("source_rows", "passed_gates", "duplicates", "inserted")}}])
    _append(log, "ingest_log", u)
    return summary


def _append(frame: pd.DataFrame, table: str, url: str) -> None:
    if is_postgres(url):
        frame.to_sql(table, _engine(url), if_exists="append", index=False,
                     method="multi", chunksize=500)
        return
    p = sqlite_path(url)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    try:
        frame.to_sql(table, con, if_exists="append", index=False)
    finally:
        con.close()
