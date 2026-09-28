"""Loaders for the INRB/UMIE BDBV2026-Data repository.

All loaders resolve health-zone names to the canonical ``nom`` used by the
shapefile, applying ``data/aliases.csv`` so that spelling variants
(Mongbwalu/Mongbalu, Nyankunde/Nyakunde, ...) collapse onto one zone.

The data lives in a sibling clone at ``data/BDBV2026-Data`` and is NOT tracked
in this repo (see DATA.md). Set ``BDBV_DATA_DIR`` to override the location.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .model import to_day


def data_root() -> Path:
    """Return the BDBV2026-Data clone root, honouring ``BDBV_DATA_DIR``.

    Walks up from this file until it finds the repo root, identified by
    the presence of `data/BDBV2026-Data`, so the model package can be
    relocated without breaking data discovery.
    """
    env = os.environ.get("BDBV_DATA_DIR")
    if env:
        root = Path(env)
    else:
        p = Path(__file__).resolve()
        # walk up looking for data/BDBV2026-Data
        for anc in p.parents:
            candidate = anc / "data" / "BDBV2026-Data"
            if candidate.exists():
                root = candidate
                break
        else:
            # fallback if walk found nothing (shouldn't happen from a normal checkout)
            root = Path.cwd() / "data" / "BDBV2026-Data"
    if not root.exists():
        raise FileNotFoundError(
            f"BDBV2026-Data not found at {root}. Clone it (see DATA.md) or set BDBV_DATA_DIR."
        )
    return root


@cache
def alias_map() -> dict[str, str]:
    """observed_name -> canonical_nom, from ``data/aliases.csv``."""
    df = pd.read_csv(data_root() / "data" / "aliases.csv")
    return dict(zip(df["observed_name"], df["canonical_nom"]))


def canonicalize(names: pd.Series) -> pd.Series:
    """Map a Series of observed zone names to canonical ``nom``."""
    return names.replace(alias_map())


def _processed(dataset: str, filename: str) -> Path:
    return data_root() / "data" / dataset / "processed" / filename


def _parse_date_col(s: pd.Series, source: str = "") -> pd.Series:
    """Parse a date column, tolerating stray bracket characters seen in some
    INRB SitRep CSV exports (e.g. ``2026-06-25]``). Any date that still fails
    to parse after stripping brackets is coerced to NaT and reported, rather
    than crashing the whole loader on one bad row."""
    cleaned = s.astype(str).str.replace(r"[\[\]]", "", regex=True)
    parsed = pd.to_datetime(cleaned, errors="coerce")
    n_bad = int(parsed.isna().sum() - s.isna().sum())
    if n_bad > 0:
        print(f"[data.py] WARNING: {n_bad} unparseable date(s) in {source or 'input'}, dropped")
    return parsed


def load_confirmed_cases_daily(drop_unattributed: bool = False) -> pd.DataFrame:
    """Cumulative confirmed cases by zone and date (the richest temporal signal).

    Columns: ``nom``, ``date`` (datetime), ``cumulative_confirmed_cases``.
    Zone names are canonicalized and re-aggregated so alias variants merge.
    The unattributed ``NA`` zone is retained by default and flagged via
    ``drop_unattributed`` — it is a material share of the national total.
    """
    # keep_default_na=False so the literal "NA" zone (unattributed cases, a
    # material share of the national total) is NOT parsed as a missing value.
    df = pd.read_csv(
        _processed("insp_sitrep", "insp_sitrep__cumulative_confirmed_cases__daily.csv"),
        keep_default_na=False,
    )
    df["date"] = _parse_date_col(df["date"])
    df["nom"] = canonicalize(df["nom"])
    # The source uses "ND" (non disponible) for not-reported; coerce to NaN
    # so it reads as missing rather than poisoning the column to string dtype.
    # With keep_default_na=False, empty cells are "" and also coerce to NaN.
    df["cumulative_confirmed_cases"] = pd.to_numeric(
        df["cumulative_confirmed_cases"], errors="coerce"
    )
    if drop_unattributed:
        df = df[df["nom"] != "NA"]
    # Re-aggregate after alias merge (two spellings on the same date collapse).
    # min_count=1 keeps all-missing groups as NaN rather than summing to 0.
    df = (
        df.groupby(["nom", "date"], as_index=False)["cumulative_confirmed_cases"]
        .sum(min_count=1)
        .sort_values(["nom", "date"])
        .reset_index(drop=True)
    )
    return df


def load_national_confirmed_daily() -> pd.DataFrame:
    """National cumulative confirmed cases by date. Columns: ``date``, value."""
    df = pd.read_csv(
        _processed(
            "insp_sitrep",
            "insp_sitrep__national_cumulative_confirmed_cases__daily.csv",
        )
    )
    df["date"] = _parse_date_col(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_national_deaths_daily() -> pd.DataFrame:
    """National cumulative confirmed deaths by date. Columns: ``date``, value."""
    df = pd.read_csv(
        _processed(
            "insp_sitrep",
            "insp_sitrep__national_cumulative_confirmed_deaths__daily.csv",
        )
    )
    df["date"] = _parse_date_col(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_suspects_in_isolation_daily() -> pd.DataFrame:
    """National count of suspected cases currently in isolation, by date.
    Columns: ``date``, ``value``. A daily census (not cumulative)."""
    df = pd.read_csv(
        _processed("insp_sitrep", "insp_sitrep__national_suspected_cases_in_isolation__daily.csv")
    )
    df = df.rename(columns={"national_suspected_cases_in_isolation": "value"})
    df["date"] = _parse_date_col(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)[["date", "value"]]
    )


def load_isolation_stockflow_validation(drop_n18: bool = True) -> pd.DataFrame:
    """Reconciled national daily isolation stock-flow series from the SitRep-PDF
    Claude-analysis xlsx (data/inputs/MVE_isolation_stockflow_validation.xlsx,
    sheet 'National Reconciliation'). Covers N°18 (2026-06-01) → N°39 (2026-06-22) with N°29 (06-12) and
    N°35 (06-18) missing in the source.

    Returns columns: ``date``, ``opening_stock``, ``admissions``, ``sorties``,
    ``total_close``, ``confirmed``, ``suspect``.

    The ``confirmed`` series is the model's CLEAN tracing comparator (true cases ↔ true
    cases, no positivity scaling); ``suspect`` and the flows are PPC overlays.

    N°19→N°39 reconcile exactly on both identities; **N°18 (2026-06-01) is the single
    exception** (flow residual +6, composition residual +4 nationally; "158/171 in one
    place vs 173 in the table" — internal SitRep inconsistency, not parsing). Dropped by
    default; pass ``drop_n18=False`` for diagnostics. Closing-only — no opening-confirmed
    series; do NOT chain across reports (re-basing on ~80% of pairs)."""
    repo_data = data_root().parent
    fp = repo_data / "inputs" / "MVE_isolation_stockflow_validation.xlsx"
    df = pd.read_excel(fp, sheet_name="National Reconciliation", header=3)
    numeric_cols = [
        "opening_stock",
        "admissions",
        "sorties",
        "total_close",
        "confirmed",
        "suspect",
    ]
    df = df.rename(
        columns={
            "Reporting date": "date",
            "Au lit (J-1)": "opening_stock",
            "Admissions (24h)": "admissions",
            "Total sorties": "sorties",
            "Reported isolement": "total_close",
            "Confirmés (NC+AC)": "confirmed",
            "Suspects": "suspect",
        }
    )
    df = df[["date", *numeric_cols]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["confirmed"]).sort_values("date").reset_index(drop=True)
    if drop_n18:
        df = df[df["date"] != pd.Timestamp("2026-06-01")].reset_index(drop=True)
    return df


def load_recovered_national_daily() -> pd.DataFrame:
    """National cumulative recovered cases (cumul guéris) by date.
    Per the SitRep methods: confirmed cases recorded as recovered (released after the control-swab
    cycle). Daily series 2026-05-27 → 2026-06-22 (26 points, 1 → 115 cumulative). Clean comparator
    against the model's `detected & ~died & iso_end ≤ d` (confirmed survivors discharged from iso).
    Returns columns ``date``, ``value`` (cumulative recovered)."""
    df = pd.read_csv(
        _processed("insp_sitrep", "insp_sitrep__national_cumulative_recovered_cases__daily.csv")
    )
    df = df.rename(columns={"national_cumulative_recovered_cases": "value"})
    df["date"] = _parse_date_col(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)[["date", "value"]]
    )


def load_new_hosp_admissions_national_daily() -> pd.DataFrame:
    """National daily NEW admissions to isolation (a flow signal): sum of per-zone
    new_all_admissions across zones; sparse coverage (~6 zones, ~mid-May to mid-June). Returns
    columns ``date``, ``value``. Flow signal (NOT cumulative) — circumvents the bed-stay-duration
    problem inherent in the suspects-in-isolation STOCK census."""
    df = pd.read_csv(_processed("insp_sitrep", "insp_sitrep__new_hosp_admissions__daily.csv"))
    df["date"] = _parse_date_col(df["date"])
    df["new_all_admissions"] = pd.to_numeric(df["new_all_admissions"], errors="coerce")
    nat = (
        df.groupby("date")["new_all_admissions"]
        .sum(min_count=1)
        .dropna()
        .reset_index()
        .rename(columns={"new_all_admissions": "value"})
    )
    return nat.sort_values("date").reset_index(drop=True)


def load_contacts_traced_national_daily() -> pd.DataFrame:
    """National cumulative contacts traced, by date. Built by
    forward-filling each zone's cumulative series over ND gaps and summing across
    zones per date. Columns: ``date``, ``value``."""
    df = pd.read_csv(
        _processed("insp_sitrep", "insp_sitrep__cumulative_contacts_traced__daily.csv")
    )
    df["date"] = _parse_date_col(df["date"])
    df["cumulative_contacts_traced"] = pd.to_numeric(
        df["cumulative_contacts_traced"], errors="coerce"
    )
    wide = (
        df.pivot_table(
            index="date", columns="nom", values="cumulative_contacts_traced", aggfunc="max"
        )
        .sort_index()
        .ffill()
    )
    nat = wide.sum(axis=1).reset_index()
    nat.columns = ["date", "value"]
    return nat


def load_pcr_machines() -> pd.DataFrame:
    """PCR machine counts by zone. Columns: ``nom``, ``pcr_machines``."""
    df = pd.read_csv(_processed("testing_capacity", "testing_capacity__pcr_machines__static.csv"))
    df["nom"] = canonicalize(df["nom"])
    return df.groupby("nom", as_index=False)["pcr_machines"].sum()


def load_healthsite_counts() -> pd.DataFrame:
    """Health-facility counts by zone. Columns: ``nom``, ``healthsite_count``."""
    df = pd.read_csv(
        _processed(
            "grid3_healthsites",
            "grid3_healthsites__healthsite_count__static.csv",
        )
    )
    df["nom"] = canonicalize(df["nom"])
    value_col = next(c for c in df.columns if c != "nom")
    return df.rename(columns={value_col: "healthsite_count"})[["nom", "healthsite_count"]]


def load_mobility_outflow(source: str = "pdf") -> pd.DataFrame:
    """Full DRC Flowminder outflow matrix, square DataFrame indexed by ``nom``.

    ``M.loc[origin, dest]`` is the trip count from origin to dest. Row and
    column names are canonicalized; duplicate names are summed.

    ``source="pdf"`` = the sitrep-PDF-extracted static matrix.
    ``source="hdx"`` (used here) = the raw HDX Flowminder national OD: broader
    coverage, identical where the two overlap. See data/DATA.md.
    """
    if source == "hdx":
        repo_data = data_root().parent
        df = pd.read_csv(
            repo_data
            / "flowminder_hdx"
            / "processed"
            / "flowminder_hdx__outflow_2026_03.matrix.csv",
            index_col=0,
        )
    else:
        df = pd.read_csv(
            _processed("flowminder", "flowminder__outflow__static.matrix.csv"),
            index_col="nom",
        )
    df.index = canonicalize(pd.Series(df.index)).to_numpy()
    df.columns = canonicalize(pd.Series(df.columns)).to_numpy()
    df = df.groupby(level=0).sum().T.groupby(level=0).sum().T
    return df


def load_travel_time_matrix() -> pd.DataFrame:
    """OSRM road travel-time matrix in minutes, square, canonical ``nom`` index.

    ``M.loc[a, b]`` is road minutes from a to b. Unroutable pairs are NaN.
    """
    df = pd.read_csv(
        _processed("osrm", "osrm__travel_time__static.matrix.csv"),
        index_col="nom",
    )
    df.index = canonicalize(pd.Series(df.index)).to_numpy()
    df.columns = canonicalize(pd.Series(df.columns)).to_numpy()
    df = df.groupby(level=0).min().T.groupby(level=0).min().T
    return df


def load_lab_schedule() -> list[dict]:
    """Local PCR labs from ``data/inputs/lab_rollout.yaml`` (Kinshasa baseline excluded).

    Returns dicts with canonical ``zone_nom``, ``online_date`` (Timestamp),
    ``online_day`` (int day from ``model.REF_DATE``, for the integer-day
    bookkeeping the simulator uses), and ``location``. Labs without a firm
    online date are skipped.
    """
    import yaml

    spec = yaml.safe_load((data_root().parent / "inputs" / "lab_rollout.yaml").read_text())
    amap = alias_map()
    labs = []
    for lab in spec["labs"]:
        if lab["zone_nom"] == "Kinshasa":
            continue  # the ship-in baseline, not a local lab
        if not lab.get("online_date"):
            continue
        online_date = pd.Timestamp(lab["online_date"])
        labs.append(
            {
                "zone_nom": amap.get(lab["zone_nom"], lab["zone_nom"]),
                "online_date": online_date,
                "online_day": to_day(online_date),
                "location": lab["location"],
            }
        )
    return labs


def load_health_zones() -> gpd.GeoDataFrame:
    """Health-zone boundaries with merged per-zone attributes.

    Returns a GeoDataFrame keyed on ``nom`` (already canonical in the build).
    """
    gdf = gpd.read_file(data_root() / "build" / "drc_health_zones.geojson")
    return gdf
