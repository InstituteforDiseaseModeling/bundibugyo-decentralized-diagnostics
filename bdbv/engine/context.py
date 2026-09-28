"""Build the shared calibration context: zone set, observed series, mobility, lab
rollout, care referral, priors and fixed values, packed into the dict workers receive."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..model import data
from ..model.model import build_served_from_day, from_day, to_day
from ..model.priors import gamma_shape_scale, lognormal_params
from .simulation import discrete_gi, dm_lump
from .state import CORE

# The bdbv/ package directory, where the static inputs shipped with the model live.
BDBV_DIR = Path(__file__).resolve().parents[1]


def build_zone_set(case_zones, mobility, cfg):
    core = [z for z in CORE if z in mobility.index]
    inflow = mobility.loc[core].sum(axis=0).sort_values(ascending=False)
    nbrs = [z for z in inflow.index if z not in case_zones][
        : cfg["scope"]["n_frontier_neighbours"]
    ]
    zones = (
        set(case_zones)
        | set(nbrs)
        | set(cfg["scope"]["always_include"])
        | {cfg["seed_infection"]["zone"]}
    )
    return sorted(zones)


def _listing_load(nat: pd.DataFrame, day_ints: np.ndarray, window: float) -> tuple:
    """Observed national confirmations in the trailing `window` days, per day.

    This is the DEMAND on the contact-listing queue: the response lists contacts for the cases
    it confirms, at a throughput that the SitReps show is flat in incidence (435/day, r = 0.13
    against same-day cases). Returned as (days, vals) for `model.simulate(listing_load=...)`.

    Reported to the log so the number driving the mechanism is never implicit.
    """
    val = next(c for c in nat.columns if c not in ("date", "nom"))
    s = nat.set_index("date")[val].astype(float)
    s = s[~s.index.duplicated()].sort_index().asfreq("D").interpolate()
    cum = np.array(
        [
            float(s.reindex([from_day(int(t))]).ffill().iloc[0])
            if from_day(int(t)) >= s.index[0]
            else 0.0
            for t in day_ints
        ]
    )
    w = round(window)
    load = cum - np.concatenate([np.zeros(min(w, len(cum))), cum[:-w]])[: len(cum)]
    load = np.maximum(load, 0.0)
    print(
        f"[setup] exogenous listing load (obs confirmations per {w} d): "
        f"{load[0]:.0f} at {from_day(int(day_ints[0])).date()} -> "
        f"{load[-1]:.0f} at {from_day(int(day_ints[-1])).date()}; "
        f"max {load.max():.0f}",
        flush=True,
    )
    return (day_ints.astype(float), load)


def build_context(cfg) -> dict:
    cc = data.load_confirmed_cases_daily()
    ccz = cc[cc["nom"] != "NA"].copy()
    nat = data.load_national_confirmed_daily()
    deaths = data.load_national_deaths_daily()
    iso_stock = data.load_suspects_in_isolation_daily()  # overlay (raw national susp)
    isoval = data.load_isolation_stockflow_validation()  # SUMMARY STAT 5 (confirmed-in-iso, clean)
    adm = (
        data.load_new_hosp_admissions_national_daily()
    )  # legacy overlay; admissions stream below comes from isoval
    recovered = data.load_recovered_national_daily()  # NEW summary stat 7 (clean recovered)
    contacts = data.load_contacts_traced_national_daily()  # overlay
    mobility = data.load_mobility_outflow(source=cfg.get("mobility_source", "pdf"))
    travel = data.load_travel_time_matrix()
    labs = data.load_lab_schedule()
    horizon_date = pd.Timestamp(cfg["horizon_date"])
    dates = pd.date_range(min(cc["date"].min(), pd.Timestamp("2026-05-13")), horizon_date)
    day_ints = np.array([to_day(d) for d in dates])
    cutoff_day = to_day(cfg["horizon_date"])
    response_day = to_day(cfg["response_start_date"])
    case_zones = sorted(ccz["nom"].unique())
    # The FULL national footprint, not a reduced outbreak-core subset. This eliminates the
    # scope mismatch between calibration and forecast, so a posterior can be replayed
    # bit-identically in the intervention sim. Downstream code (DM top-15 lump,
    # care_destination, tier lists) operates on this expanded set.
    zones = sorted(set(mobility.index) | set(case_zones) | {cfg["seed_infection"]["zone"]})
    Z = len(zones)
    zi = {z: i for i, z in enumerate(zones)}
    print(f"[setup] national footprint: Z={Z} health zones", flush=True)
    # Per-zone transmission-heterogeneity score in [0,1] (0 = high-disruption reference
    # class, transmits at the calibrated R; 1 = low-disruption province, transmits at
    # rho*R). The tracked CSV carries one suitably transformed ACLED-derived score per
    # province; raw ACLED events and intermediate rates are not distributed.
    # The model multiplier is m_z = 1 - (1 - rho) * s_z, so s = 0 everywhere (or rho = 1)
    # removes the heterogeneity entirely.
    _cov = pd.read_csv(BDBV_DIR / cfg.get("province_covariates", "province_covariates.csv"))
    _s_col = cfg.get("zone_hetero_column", "s_disruption")
    if _s_col not in _cov.columns:
        raise KeyError(
            f"zone_hetero_column {_s_col!r} not in province_covariates.csv "
            f"(have: {[c for c in _cov.columns if c.startswith('s_')]})"
        )
    _s_by_province = dict(zip(_cov["province"], _cov[_s_col].astype(float)))

    _hz = data.load_health_zones()[["nom", "province"]]
    _s_map = {
        str(row.nom): _s_by_province[row.province]
        for row in _hz.itertuples(index=False)
        if row.province in _s_by_province
    }
    # The build geojson disambiguates a few duplicated zone names with a province suffix
    # ("Bili (Bas-Uele)"); the model's zone set uses the bare name. Match on the stem too.
    for _n, _s in list(_s_map.items()):
        _stem = _n.split(" (")[0]
        _s_map.setdefault(_stem, _s)
    _missing_cov = [z for z in zones if z not in _s_map]
    if _missing_cov:
        print(
            f"[setup] WARNING {len(_missing_cov)} zones absent from province_covariates.csv "
            f"-> s_hetero = 1.0 (no organised violence): {_missing_cov[:10]}",
            flush=True,
        )
    s_hetero = np.array([_s_map.get(z, 1.0) for z in zones], dtype=float)
    print(
        f"[setup] zone heterogeneity from {_s_col!r}: {int((s_hetero < 0.5).sum())} of {Z} "
        f"zones in the reference class (s<0.5); mean s = {s_hetero.mean():.3f}",
        flush=True,
    )
    M = np.nan_to_num(mobility.reindex(index=zones, columns=zones).to_numpy().astype(float))
    present = np.array([z in mobility.index for z in zones])
    # keep the nearest-neighbor COLUMN copy so tier-3 zones stay reachable as
    # destinations from tier-1/2 (their inbound weight is preserved). Their
    # OUTBOUND row is overwritten with pure gravity per-draw in `three_tier_mix`, replacing
    # the (now-column-only) copy-neighbor hack.
    for mi in np.where(~present)[0]:
        z = zones[mi]
        cand = [
            (travel.at[z, zones[k]], k)
            for k in range(Z)
            if present[k]
            and z in travel.index
            and zones[k] in travel.columns
            and np.isfinite(travel.at[z, zones[k]])
        ]
        if cand:
            _, j = min(cand)
            M[:, mi] = M[:, j]  # column-only copy (was: also row)
    np.fill_diagonal(M, 0.0)
    rs = M.sum(1, keepdims=True)
    M = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    tier3_idx = np.where(~present)[0].astype(int)  # Flowminder-missing zones
    tier3_names = [zones[i] for i in tier3_idx]
    print(
        f"[setup] tier 3 (Flowminder-missing → pure gravity outbound): {tier3_names}", flush=True
    )
    sched = [
        {"zone_nom": l["zone_nom"], "online_day": to_day(l["online_date"])}
        for l in labs
        if l["zone_nom"] in travel.index
    ]
    # The lab road-catchment radius lives in `fixed:`, and only there. Two mechanical guards,
    # because this single number sets both the status-quo turnaround tier and the
    # placement-eligible zone set:
    #   (a) `scope.catchment_minutes` must not exist — with the radius readable from two
    #       places, one of them silently wins;
    #   (b) it has no default, so a config that omits it raises here rather than falling
    #       back to some other radius.
    if "catchment_minutes" in cfg["scope"]:
        raise ValueError(
            "scope.catchment_minutes is set — the catchment radius belongs in "
            "fixed:; with both present one would silently win. Delete it."
        )
    _catch = cfg["fixed"].get("catchment_minutes")
    if _catch is None:
        raise ValueError(
            "fixed.catchment_minutes is missing — set the catchment radius in minutes"
        )
    served = build_served_from_day(zones, travel, sched, float(_catch))
    collect_high = served.copy()
    # observed zone-final (cases)
    obs_conf = (
        ccz.pivot_table(
            index="date", columns="nom", values="cumulative_confirmed_cases", aggfunc="sum"
        )
        .reindex(dates)
        .reindex(columns=zones)
        .ffill()
        .fillna(0)
    )
    obs_zone_final = obs_conf.iloc[-1].to_numpy()
    # observed per-zone cumulative deaths at horizon (for deaths spatial DM)
    _deaths_zone_df = pd.read_csv(
        data.data_root()
        / "data"
        / "insp_sitrep"
        / "processed"
        / "insp_sitrep__cumulative_confirmed_deaths__daily.csv"
    )
    _deaths_zone_df["date"] = pd.to_datetime(_deaths_zone_df["date"], errors="coerce")
    _deaths_zone_df["count"] = pd.to_numeric(
        _deaths_zone_df["cumulative_confirmed_deaths"], errors="coerce"
    )
    _deaths_zone_df["nom"] = _deaths_zone_df["nom"].map(lambda s: data.alias_map().get(s, s))
    _deaths_zone_df = _deaths_zone_df.groupby(["nom", "date"], as_index=False).agg(
        count=("count", "sum")
    )
    obs_deaths_pv = (
        _deaths_zone_df.pivot_table(index="date", columns="nom", values="count", aggfunc="sum")
        .reindex(dates)
        .reindex(columns=zones)
        .ffill()
        .fillna(0)
    )
    obs_zone_deaths_final = obs_deaths_pv.iloc[-1].to_numpy()

    # precompute DM lumping partition. Top-N zones by obs_zone_final are
    # scored as explicit categories; the rest lump into one "other" bucket. Same
    # partition used for cases and deaths so top-N is determined once by
    # observed cases magnitude.
    _dm_top_n = int(cfg.get("dm_top_n", 15))
    _order = np.argsort(-obs_zone_final)  # descending obs
    dm_top_idx = _order[:_dm_top_n].astype(int)
    dm_other_mask = np.ones(len(zones), dtype=bool)
    dm_other_mask[dm_top_idx] = False
    obs_zone_final_lumped = dm_lump(obs_zone_final, dm_top_idx, dm_other_mask)
    obs_zone_deaths_final_lumped = dm_lump(obs_zone_deaths_final, dm_top_idx, dm_other_mask)
    _top_names = [zones[i] for i in dm_top_idx]
    _other_cases = int(obs_zone_final_lumped[-1])
    _other_deaths = int(obs_zone_deaths_final_lumped[-1])
    print(f"[setup] DM top-{_dm_top_n} zones: {_top_names}", flush=True)
    print(
        f"[setup] DM 'other' bucket: {int(dm_other_mask.sum())} zones, "
        f"cases={_other_cases} ({100 * _other_cases / max(int(obs_zone_final.sum()), 1):.1f}%), "
        f"deaths={_other_deaths} ({100 * _other_deaths / max(int(obs_zone_deaths_final.sum()), 1):.1f}%)",
        flush=True,
    )

    # build care_destination map from care_facilities.yaml + travel-time matrix
    _cf_path = BDBV_DIR / "model" / "care_facilities.yaml"
    _cf = yaml.safe_load(_cf_path.read_text())
    _cte_zones = set(_cf["cte_zones"])
    _cte_indices = [i for i, z in enumerate(zones) if z in _cte_zones]
    care_destination = np.arange(len(zones), dtype=np.int64)  # default: self (identity)
    for i, z in enumerate(zones):
        if z in _cte_zones:
            continue  # CTE stays self
        # find nearest CTE by travel-time (fallback: any CTE if no travel-time; last-resort: keep self)
        best_j = None
        best_tt = np.inf
        for j in _cte_indices:
            cte_z = zones[j]
            if z in travel.columns and cte_z in travel.index:
                tt = travel.at[cte_z, z]
                if pd.notna(tt) and tt < best_tt:
                    best_tt = tt
                    best_j = j
        if best_j is not None:
            care_destination[i] = best_j
    _n_referring = int((care_destination != np.arange(len(zones))).sum())
    print(
        f"[setup] care_destination: {len(_cte_indices)} CTE zones ({sorted(_cte_zones & set(zones))}); "
        f"{_n_referring} referring zones → mapped to nearest CTE",
        flush=True,
    )
    # observed national cumulative confirmed (for daily cumulative PPC)
    obs_nat_cum = (
        nat.set_index("date")["national_cumulative_confirmed_cases"]
        .reindex(dates)
        .ffill()
        .fillna(0)
        .to_numpy()
    )
    obs_deaths_cum = (
        deaths.set_index("date")["national_cumulative_confirmed_deaths"]
        .reindex(dates)
        .ffill()
        .fillna(0)
        .to_numpy()
    )
    # confirmed summary edges (3-day-new): edges day_ints[0], +step, ..., last
    step = int(cfg["confirmed_step_days"])
    conf_edges = list(range(int(day_ints[0]), int(day_ints[-1]) + 1, step))
    if conf_edges[-1] != int(day_ints[-1]):
        conf_edges.append(int(day_ints[-1]))
    conf_edges = np.asarray(conf_edges, int)
    # observed 3-day-new confirmed from cumulative at the edges
    obs_conf_new = np.diff(np.interp(conf_edges, day_ints, obs_nat_cum))
    # death edges = two-weekly (14-day) NB cells laid out
    # BACKWARD from `lower_bound_start_date` (so that date lands exactly on an edge
    # and cells are exactly bin_step_days wide; the oldest cell — near-zero deaths —
    # absorbs the remainder), then ONE terminal cell from that date to the horizon
    # scored as a soft lower bound. `n_death_nb_cells` = the leading NB cells; the
    # final diff (death_edges[-2] -> death_edges[-1]) is the lower-bound cell.
    _dl = cfg.get("death_likelihood", {})
    _death_step = int(_dl.get("bin_step_days", 14))
    _lb_start_day = to_day(_dl.get("lower_bound_start_date", "2026-06-29"))
    _d_start = int(day_ints[0])
    _d_horizon = int(day_ints[-1])
    _back = list(
        range(_lb_start_day, _d_start, -_death_step)
    )  # lb_start, lb_start-step, ... (> start)
    death_edges = sorted(_back)
    if not death_edges or death_edges[0] > _d_start:
        death_edges = [_d_start] + death_edges
    death_edges = np.asarray(death_edges + [_d_horizon], int)
    n_death_nb_cells = len(death_edges) - 2  # leading NB cells; last cell is the terminal cell
    obs_deaths_binned = np.diff(np.interp(death_edges, day_ints, obs_deaths_cum))
    # terminal cell mode. "lower_bound" = asymmetric (over-prediction
    # free); "two_sided" = neutral symmetric NB against obs x terminal_completeness.
    _term_mode = str(_dl.get("terminal_mode", "lower_bound"))
    _term_completeness = float(_dl.get("terminal_completeness", 1.0))
    _term_disp = float(_dl.get("terminal_disp", cfg["nb_dispersion"]))
    obs_deaths_terminal = obs_deaths_binned[n_death_nb_cells:] / _term_completeness
    print(
        f"[setup] death rail: {n_death_nb_cells} two-weekly NB cells + 1 terminal "
        f"cell [{_lb_start_day}->{_d_horizon}] scored {_term_mode.upper()}; obs incident "
        f"deaths per cell = {np.round(obs_deaths_binned).astype(int).tolist()} (terminal "
        f"target = {obs_deaths_terminal[-1]:.0f}, completeness {_term_completeness:g}, "
        f"disp {_term_disp:g} vs {cfg['nb_dispersion']:g} elsewhere)",
        flush=True,
    )
    # Backlog: scored anchor(s) + optional non-scored overlay anchor(s).
    bk = cfg["backlog_targets"]
    scored_days = [to_day(d) for d in bk["dates"]]
    obs_backlog = np.asarray(bk["values"], float)
    # fixed positivity scalar for backlog target (203 × 0.35 = 71 true-positive equivalent).
    backlog_positivity_scalar = float(cfg.get("backlog_positivity_scalar", 0.35))
    obs_backlog_scaled = np.maximum(
        np.round(obs_backlog * backlog_positivity_scalar).astype(int), 1
    )
    overlay_dates = cfg.get("backlog_overlay_dates", [])
    overlay_values = np.asarray(cfg.get("backlog_overlay_values", []), float)
    overlay_days = [to_day(d) for d in overlay_dates]
    # backlog_query_days = scored followed by overlay (so positions 0..n_scored-1 go into the NB term)
    backlog_query_days = np.array(scored_days + overlay_days, int)
    n_scored_backlog = len(scored_days)
    # confirmed-in-iso summary stat: SitRep-validated, daily, 06-01 -> 06-22 (19 reconciled + 1 flagged).
    # Treated as point observations through the NB obs model — no differencing (points don't chain).
    isoval = isoval[isoval["date"] <= horizon_date]
    isoval_day_ints = np.array([to_day(d) for d in isoval["date"]], int)
    obs_confirmed_iso = isoval["confirmed"].to_numpy()
    obs_suspect_iso_val = isoval["suspect"].to_numpy()  # overlay
    # summary stat 6: 3-day-new admissions from xlsx (daily Admissions (24h), binned).
    # Edges = isoval dates binned in 3-day windows starting at the first isoval date.
    isoval_min_day = int(isoval_day_ints.min())
    isoval_max_day = int(isoval_day_ints.max())
    adm_edges = list(range(isoval_min_day, isoval_max_day + 1, 3))
    if adm_edges[-1] != isoval_max_day:
        adm_edges.append(isoval_max_day)
    adm_edges = np.asarray(adm_edges, int)
    # observed binned admissions = sum within each window from the xlsx daily flow
    obs_adm_daily = isoval["admissions"].to_numpy()
    obs_adm_new = np.zeros(len(adm_edges) - 1, float)
    for k in range(len(adm_edges) - 1):
        d0, d1 = adm_edges[k], adm_edges[k + 1]
        # cells with day in (d0, d1] (left-exclusive, right-inclusive to match np.diff semantics)
        days_in_bin = [i for i, dd in enumerate(isoval_day_ints) if d0 < dd <= d1]
        obs_adm_new[k] = obs_adm_daily[days_in_bin].sum() if days_in_bin else 0.0
    # summary stat 7: 3-day-new recovered from the national guéris series.
    recovered = recovered[recovered["date"] <= horizon_date]
    rec_day_ints = np.array([to_day(d) for d in recovered["date"]], int)
    rec_cum = recovered["value"].to_numpy()
    # Edges aligned to the recovered window (3-day step)
    rec_min_day = int(rec_day_ints.min())
    rec_max_day = int(rec_day_ints.max())
    rec_edges = list(range(rec_min_day, rec_max_day + 1, 3))
    if rec_edges[-1] != rec_max_day:
        rec_edges.append(rec_max_day)
    rec_edges = np.asarray(rec_edges, int)
    # Observed binned recovered = diff of cumulative interpolated at edges
    obs_rec_new = np.diff(np.interp(rec_edges, rec_day_ints, rec_cum))
    # PPC stock (raw national susp series — kept for full-window overlay)
    iso_stock = iso_stock[iso_stock["date"] <= horizon_date]
    gi_shape, gi_scale = gamma_shape_scale(cfg["fixed"]["gi_mean"], cfg["fixed"]["gi_sd"])
    incub_ml, incub_sl = lognormal_params(cfg["fixed"]["incub_mean"], cfg["fixed"]["incub_sd"])
    ccap = cfg.get("confirm_capacity")
    # cap_vals may contain `null` for entries calibrated per-draw (cap_phei).
    # Convert None → NaN; simulate_draw fills the calibrated value from p["cap_phei"] per draw.
    cap_vals_raw = [
        float("nan") if v is None else float(v) for v in (ccap["caps"] if ccap else [])
    ]
    cap = (
        None
        if ccap is None
        else (np.array([to_day(d) for d in ccap["dates"]]), np.array(cap_vals_raw, float))
    )
    # --- precompute pairwise HZ distances + destination-pop numerator for the gravity kernel ---
    gdf = data.load_health_zones()
    _pts = {
        row["nom"]: row.geometry.representative_point()
        for _, row in gdf.iterrows()
        if row["nom"] in set(zones)
    }
    _lat = np.radians(np.array([_pts[z].y if z in _pts else np.nan for z in zones]))
    _lon = np.radians(np.array([_pts[z].x if z in _pts else np.nan for z in zones]))
    _dlat = _lat[:, None] - _lat[None, :]
    _dlon = _lon[:, None] - _lon[None, :]
    _a = (
        np.sin(_dlat / 2) ** 2
        + np.cos(_lat[:, None]) * np.cos(_lat[None, :]) * np.sin(_dlon / 2) ** 2
    )
    gravity_dist = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(_a, 0.0, 1.0)))  # (Z, Z) km
    gravity_dist = np.nan_to_num(
        gravity_dist, nan=1e6
    )  # missing centroid -> effectively-infinite dist
    wp_static = pd.read_csv(
        data.data_root() / "data" / "worldpop" / "processed" / "worldpop__pop_count__static.csv"
    ).set_index("nom")["pop_count"]
    _pop = np.array([float(wp_static.get(z, 0.0)) for z in zones])
    gravity_num_pop = np.broadcast_to(
        _pop[None, :], (Z, Z)
    ).copy()  # rows = source, cols = destination (pop_j)
    gravity_d0 = float(cfg["fixed"].get("gravity_d0_km", 5.0))
    # --- load short-trips cohort profiles (Ituri + NK) and precompute tier indices ---
    st_root = data.data_root() / "data" / "flowminder_short_trips" / "processed"

    def _load_cohort_row(fname):
        m = pd.read_csv(st_root / fname)
        v = m.iloc[0, 1:].astype(float)  # first data row (all rows are identical)
        v.index = m.columns[1:]
        row = np.array([float(v.get(z, 0.0)) for z in zones])
        s = row.sum()
        return row / s if s > 0 else row

    st_ituri = _load_cohort_row(
        "flowminder_short_trips__ituri_subscriber_days_followup_20260608__static.matrix.csv"
    )
    st_nk = _load_cohort_row(
        "flowminder_short_trips__nk_subscriber_days_followup_20260608__static.matrix.csv"
    )
    # Tier-1 origin lists (subset that's actually in the calibrated `zones` set)
    tier1_ituri_names = [
        z for z in ["Bunia", "Mongbwalu", "Rwampara", "Nyankunde"] if z in set(zones)
    ]  # FIX canonical spellings
    tier1_nk_names = [z for z in ["Beni", "Butembo", "Katwa"] if z in set(zones)]
    tier1_ituri_idx = np.array([zi[z] for z in tier1_ituri_names], dtype=int)
    tier1_nk_idx = np.array([zi[z] for z in tier1_nk_names], dtype=int)
    print(
        f"[setup] tier 1 Ituri origins: {tier1_ituri_names}  |  NK origins: {tier1_nk_names}",
        flush=True,
    )
    print(
        f"[setup] short-trips Ituri non-zero destinations: {(st_ituri > 0).sum()}   NK: {(st_nk > 0).sum()}",
        flush=True,
    )
    shared = {
        "zones": zones,
        "M": M,
        "served": served,
        "Z": Z,
        "day_ints": day_ints,
        "cutoff_day": cutoff_day,
        "gravity_dist": gravity_dist,
        "gravity_num_pop": gravity_num_pop,
        "gravity_d0": gravity_d0,  # gravity kernel (β FIXED)
        "st_ituri": st_ituri,
        "st_nk": st_nk,  # short-trips cohort rows (row-normed)
        "tier1_ituri_idx": tier1_ituri_idx,
        "tier1_nk_idx": tier1_nk_idx,
        "tier3_idx": tier3_idx,  # tier assignments
        "care_destination": care_destination,  # zone -> CTE zone map (self if CTE)
        "obs_zone_deaths_final": obs_zone_deaths_final,  # for deaths spatial DM (unlumped, used by figs)
        "obs_zone_final_lumped": obs_zone_final_lumped,  # DM top-N + other
        "obs_zone_deaths_final_lumped": obs_zone_deaths_final_lumped,  # DM top-N + other (same partition)
        "dm_top_idx": dm_top_idx,
        "dm_other_mask": dm_other_mask,  # partition indices
        "dm_alpha_cases": float(
            cfg["fixed"].get("dm_alpha_cases", 30.0)
        ),  # DM concentration for cases (now WITHIN-CORE only)
        "dm_alpha_deaths": float(
            cfg["fixed"].get("dm_alpha_deaths", 10.0)
        ),  # DM concentration for deaths
        "betabin_s": float(
            cfg["fixed"].get("betabin_s", 300.0)
        ),  # core-vs-periphery Beta-Binomial concentration
        # `split_background` = the per-zone background added to the MODEL side of
        # the frac-other SPLIT's p_other_model ONLY. Using `zone_background` (0.5) here
        # instead adds ~228 phantom 'other' cases (455 zones x 0.5) against 148 real
        # observed 'other', and asymmetrically -- the obs side gets none -- which turns the
        # split from a share-constraint into a size-reward. 0.0 keeps the scored share raw
        # on both sides. The (unscored) overlay DM figures still use `zone_background`.
        "split_background": float(cfg.get("split_background", cfg.get("zone_background", 0.5))),
        # EXOGENOUS tracing-listing load. The trailing-`trace_window` count of
        # OBSERVED national confirmations, on the same day grid, held constant beyond the last
        # observation. This choice is forced: the model's own confirmations have the wrong
        # SHAPE for the job (observed L14 rises 2.83x from mid-June to the horizon, the model's
        # 1.37x and falling over the last fortnight), so an endogenous driver's coverage
        # multiplier turns back UP exactly where the collapse should be. `listing_load_source:
        # model` restores the endogenous (running, causal) load for comparison.
        "listing_load": _listing_load(nat, day_ints, float(cfg["fixed"].get("trace_window", 14.0)))
        if (
            str(cfg["fixed"].get("listing_load_source", "model")) == "observed"
            and np.isfinite(float(cfg["fixed"].get("trace_cap", np.inf)))
        )
        else None,
        "response_day": response_day,
        "obs_zone_final": obs_zone_final,
        "obs_conf_new": obs_conf_new,
        "obs_nat_cum_end": float(obs_nat_cum[-1]),  # for diagnostics that report x obs
        "obs_deaths_cum_end": float(obs_deaths_cum[-1]),  # for the projection deaths bracket
        "obs_deaths_binned": obs_deaths_binned,  # two-weekly + terminal death rail
        "n_death_nb_cells": int(n_death_nb_cells),  # leading NB cells (last is the terminal cell)
        "death_terminal_mode": _term_mode,  # "two_sided" (neutral) | "lower_bound"
        "obs_deaths_terminal": obs_deaths_terminal,  # completeness-adjusted terminal target
        "death_terminal_disp": _term_disp,  # looser NB disp for the terminal cell
        "s_hetero": s_hetero,  # (Z,) per-zone conflict-deficit score in [0,1]
        "obs_backlog": obs_backlog,
        "obs_backlog_scaled": obs_backlog_scaled,  # fixed-scalar positivity
        "backlog_positivity_scalar": backlog_positivity_scalar,
        "obs_confirmed_iso": obs_confirmed_iso,
        "obs_suspect_iso_val": obs_suspect_iso_val,
        "obs_adm_new": obs_adm_new,
        "obs_rec_new": obs_rec_new,  # admissions/recovered summary stats
        "conf_edges": conf_edges,
        "death_edges": death_edges,
        "adm_edges": adm_edges,
        "rec_edges": rec_edges,  # their bin edges
        "backlog_query_days": backlog_query_days,
        "n_scored_backlog": n_scored_backlog,
        "overlay_backlog_dates": overlay_dates,
        "overlay_backlog_values": overlay_values,
        "isoval_day_ints": isoval_day_ints,
        # suspect_positivity is FIXED, not calibrated
        "suspect_positivity": float(cfg["fixed"]["suspect_positivity"]),
        "contacts_per_index": float(cfg["contacts_per_index"]),
        "priors": cfg["priors"],
        "fx": cfg["fixed"],
        "gi_shape": gi_shape,
        "gi_scale": gi_scale,
        "incub_ml": incub_ml,
        "incub_sl": incub_sl,
        "seed_n": cfg["seed_infection"]["n"],
        "seed_zone": cfg["seed_infection"]["zone"],
        "start_mean": to_day(cfg["seed_infection"]["start_date_mean"]),
        "start_sd": cfg["seed_infection"]["start_date_sd_days"],
        "disp": cfg["nb_dispersion"],
        "background": cfg["zone_background"],
        "recog_day": response_day,
        "collect_high": collect_high,
        "cap": cap,  # collect_ratio moved to per-draw (calibrated)
        "aware_threshold": int(
            cfg.get("aware_threshold", 1)
        ),  # # confirms required to awaken a zone
        # time-varying unserved turnaround switch day (Kinshasa → regional hub)
        "unserved_switch_day": to_day(cfg["fixed"]["unserved_switch_date"]),
        # epi-joint soft cell
        "epi_joint_enabled": bool(cfg.get("epi_joint", {}).get("enabled", False)),
        "epi_estimate": int(cfg.get("epi_joint", {}).get("estimate", 2100)),
        "epi_disp": float(cfg.get("epi_joint", {}).get("disp", 25.0)),
        "epi_date": str(cfg.get("epi_joint", {}).get("date", "2026-07-06")),
        # effective-Rt anchor (epiforecasts latest R) + the discretised GI the
        # renewal estimator needs. rt_target_sd = 0 disables the term.
        "gi_disc": discrete_gi(gi_shape, gi_scale),
        "rt_target_mean": float(cfg.get("rt_target", {}).get("mean", 0.0)),
        "rt_target_sd": float(cfg.get("rt_target", {}).get("sd", 0.0)),
        "rt_window": int(cfg.get("rt_target", {}).get("window_days", 7)),
        "rt_lag": int(cfg.get("rt_target", {}).get("lag_days", 5)),
        "rt_epi_initial": tuple(cfg.get("rt_target", {}).get("epi_initial_band", (1.69, 2.09))),
        "rt_epi_latest": tuple(cfg.get("rt_target", {}).get("epi_latest_band", (1.07, 1.20))),
    }
    return {
        "dates": dates,
        "zones": zones,
        "Z": Z,
        "served": served,
        "obs_nat_cum": obs_nat_cum,
        "obs_zone_final": obs_zone_final,
        "obs_deaths_cum": obs_deaths_cum,
        "iso_stock_obs": iso_stock,
        "contacts": contacts,
        "adm_obs": adm,
        "isoval": isoval,
        "recovered": recovered,
        "rec_edges": rec_edges,
        "adm_edges": adm_edges,
        "shared": shared,
        "conf_edges": conf_edges,
    }
