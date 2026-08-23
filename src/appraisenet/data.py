"""Data access: the private listings database, the leakage-free protocol split, and a
synthetic corpus with the identical schema so the entire pipeline (and CI) runs without
the proprietary data.

The private dataset: 38,758 US used-vehicle listings collected from public marketplaces
and dealer websites during July-August 2026 (dealer and private-party sellers), VIN-decoded
specifications, deduplicated by vehicle, cleaned through documented quality gates. It is
NOT distributed with this repository; `data/README.md` documents the schema and the
collection ethics, and every command falls back to `synthetic_listings()` when the
database is absent.

Columns (table `listings`): id, price (target, USD), year, make, model, trim, mileage,
seller_type (dealer|private), condition (used|cpo), body_style, fuel_type, transmission,
drivetrain, cylinders, doors, displacement_l, engine_hp, gvwr_class, series,
electrification, adaptive_cruise, plant_country, original_price, region_state,
region_zip3, description (scrubbed free text).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import db
from .config import ProtocolCfg

CATEGORICAL = ["make", "model", "trim", "body_style", "drivetrain", "transmission", "fuel_type",
               "electrification", "gvwr_class", "series", "plant_country", "adaptive_cruise",
               "seller_type", "region_state"]
NUMERIC = ["mileage", "age", "miles_per_year", "doors", "cylinders", "engine_hp",
           "displacement_l", "original_price"]
TARGET = "log_price"


@dataclass
class Split:
    train: pd.DataFrame
    holdout: pd.DataFrame
    folds: np.ndarray          # fold id per train row
    synthetic: bool


def load_listings(allow_synthetic: bool = True, max_rows: int | None = None,
                  seed: int = 42) -> tuple[pd.DataFrame, bool]:
    frame = db.read_listings()
    if frame is None and db.is_postgres():
        raise RuntimeError("PostgreSQL is configured but holds no `listings` table yet; "
                           "load it with: appraisenet data ingest --source <listings.db|csv>")
    if frame is not None:
        df, synthetic = frame, False
    elif allow_synthetic:
        df = synthetic_listings(n=max_rows or 6000, seed=seed)
        synthetic = True
    else:
        raise FileNotFoundError("listings database not found and synthetic data not allowed")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df, synthetic


def engineer(df: pd.DataFrame, proto: ProtocolCfg) -> pd.DataFrame:
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["price"].between(proto.price_min, proto.price_max)]
    df = df[df["mileage"].notna() & (df["year"] >= proto.year_min)]
    df["age"] = (proto.current_year - df["year"]).clip(lower=0)
    df["miles_per_year"] = (df["mileage"] / df["age"].clip(lower=1)).round(0)
    df[TARGET] = np.log(df["price"])
    df["description"] = df.get("description", pd.Series(index=df.index, dtype=object)).fillna("")
    return df.reset_index(drop=True)


def protocol_split(df: pd.DataFrame, proto: ProtocolCfg) -> Split:
    """Random holdout (never touched during selection) + CV folds on the rest."""
    rng = np.random.RandomState(proto.seed)
    hold = rng.rand(len(df)) < proto.holdout_frac
    train, holdout = df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)
    folds = np.random.RandomState(proto.seed).randint(0, proto.folds, len(train))
    return Split(train=train, holdout=holdout, folds=folds, synthetic=False)


# ----------------------------------------------------------------------------------
# synthetic corpus: same schema, plausible price physics, so tests and CI are honest
# ----------------------------------------------------------------------------------
_MAKES = {
    "Toyoda": (["Corella", "Camrio", "RAVX", "Tundro"], 1.00),
    "Fjord": (["Focal", "F-100", "Escapade", "Mustango"], 0.97),
    "Chevalet": (["Malibou", "Silverida", "Equinaux"], 0.95),
    "Hondo": (["Civet", "Accardo", "CRV"], 1.03),
    "Bavaria": (["3er", "5er", "X-Trois"], 1.25),
    "Nissano": (["Sentro", "Altimo", "Rogua"], 0.92),
}
_TRIMS = [("Base", -0.06), ("Sport", 0.02), ("Limited", 0.08), ("Platinum", 0.15), (None, 0.0)]
_BODY = ["Sedan", "SUV", "Truck", "Hatchback", "Coupe"]
_STATES = ["CA", "TX", "FL", "WA", "NY"]
_GOOD = ["one owner", "clean title", "no accidents reported", "dealer maintained", "new tires"]
_BAD = ["accident reported", "rebuilt title", "former rental", "needs some work", "salvage title"]


def synthetic_listings(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    makes = list(_MAKES)
    for i in range(n):
        make = makes[rng.randint(len(makes))]
        models, brand_mult = _MAKES[make]
        model = models[rng.randint(len(models))]
        year = int(rng.randint(1995, 2027))
        age = max(2026 - year, 0)
        mileage = max(int(rng.normal(12_000 * age, 8_000)), 5)
        trim, trim_mult = _TRIMS[rng.randint(len(_TRIMS))]
        body = _BODY[hash(model) % len(_BODY)]
        seller = "dealer" if rng.rand() < 0.85 else "private"
        hp = int(np.clip(rng.normal(120 + 60 * brand_mult + (30 if body == "Truck" else 0), 40), 70, 700))
        base = 34_000 * brand_mult * (1 + trim_mult) * (1.15 if body == "Truck" else 1.0)
        value = base * np.exp(-0.11 * age) * np.exp(-0.35 * mileage / 150_000) * (0.96 if seller == "private" else 1.0)
        bad = rng.rand() < 0.10
        if bad:
            value *= rng.uniform(0.55, 0.85)
        price = int(np.clip(value * np.exp(rng.normal(0, 0.10)), 2_200, 99_000))
        phrases = list(rng.choice(_BAD if bad else _GOOD, size=2, replace=False))
        desc = f"{year} {make} {model} for sale. " + ", ".join(phrases) + "."
        rows.append(dict(
            id=i + 1, price=price, year=year, make=make, model=model, trim=trim,
            mileage=mileage, seller_type=seller, condition="used",
            body_style=body, fuel_type="Gasoline", transmission="Automatic",
            drivetrain=rng.choice(["FWD", "AWD", "4WD", "RWD"]), cylinders=int(rng.choice([4, 6, 8])),
            doors=4, displacement_l=round(float(rng.choice([1.5, 2.0, 2.5, 3.5, 5.0])), 1),
            engine_hp=hp, gvwr_class=rng.choice(["1", "1C", "1D", "2E"]), series=None,
            electrification=None, adaptive_cruise=rng.choice(["Standard", "Optional", None]),
            plant_country=rng.choice(["UNITED STATES (USA)", "JAPAN", "MEXICO"]),
            original_price=None, region_state=rng.choice(_STATES),
            region_zip3=str(rng.randint(900, 935)), description=desc,
        ))
    return pd.DataFrame(rows)
