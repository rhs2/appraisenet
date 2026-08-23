# Data

The private corpus (`listings.db`) is **not distributed**. It contains 38,758 US
used-vehicle listings collected from public marketplaces and dealer websites during
July-August 2026, VIN-decoded specifications via the free NHTSA vPIC API, deduplicated
per vehicle and cleaned through documented quality gates (price band $2,000-$100,000,
model year 1990+, no collector or exotic vehicles, placeholder prices and
not-road-ready listings removed, label-noise quarantine).

Identity was stripped before the database ever reached this project: no VINs, no listing
IDs, no seller or platform names, ZIP codes reduced to their 3-digit prefix, and free
text scrubbed of phone numbers, URLs, e-mail addresses and street addresses.

## How the corpus was curated

The data did not start clean, and the curation is as much a part of this study as the
models. Each raw listing record arrived with roughly **160 fields**: site-specific
presentation fields, duplicated spec fields in inconsistent formats, sparse columns that
existed on one source only, and seller-entered specs that contradict the VIN. The
pipeline that produced the 26-column modeling table:

1. **Field triage**: presentation-only, near-empty and redundant fields dropped; every
   surviving column normalised to one vocabulary (fuel types, transmissions, drivetrains,
   body styles).
2. **VIN-decode enrichment**: specifications re-derived from the government's free NHTSA
   vPIC decoder (batched, cached), which overrides seller-entered specs; this is where
   `gvwr_class`, `series`, `electrification`, `adaptive_cruise` and `plant_country` come
   from.
3. **Junk-price removal**: placeholder prices (the 99,999-style sentinels), zero/token
   prices, and listings whose advertised number is a *down payment* rather than a price,
   detected from the listing wording.
4. **Not-a-car removal**: parts vehicles, non-runners and "mechanic special" listings
   filtered by text rules.
5. **Protocol gates**: price band $2,000-$100,000 (no collector or exotic vehicles),
   model year 1990+, odometer present and plausible.
6. **Deduplication per vehicle**: the same car relisted, reposted or syndicated across
   sources appears once.
7. **Label-noise quarantine**: listings priced implausibly far from a preliminary
   fair-value fit (below 0.35x or above 2.5x) are quarantined rather than trained on,
   so typos and misdescribed vehicles cannot poison the target.
8. **Merge audits**: the corpus was assembled from several collection waves; every merge
   was checked for row counts, column alignment and distribution shifts before acceptance.

The same gates guard the door at serving time: `appraisenet data ingest` re-applies the
price band, year floor and required-field checks, and fingerprints every row so daily
additions can never re-introduce duplicates.

Table `listings`, one row per vehicle:

| column | meaning |
|---|---|
| price | asking price in USD (the target) |
| year, make, model, trim | vehicle identity |
| mileage | odometer, miles |
| seller_type | dealer or private |
| condition | used or cpo |
| body_style, fuel_type, transmission, drivetrain | normalised categorical specs |
| cylinders, doors, displacement_l, engine_hp | numeric specs (VIN-decoded first) |
| gvwr_class, series, electrification, adaptive_cruise, plant_country | VIN-decoded extras |
| original_price | pre-markdown price when the listing showed one |
| region_state, region_zip3 | coarse location for regional effects |
| description | scrubbed listing text |

To run the pipeline without the private data, do nothing: every command falls back to
`appraisenet.data.synthetic_listings()`, a generator with the identical schema and
plausible price physics (fictional makes and models), which is also what CI trains on.
Point `APPRAISENET_DB` in `.env` at your own database with this schema to use real data.
