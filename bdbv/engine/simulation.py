"""Parameter assembly, the simulator call, and the observation summaries it is scored on.

The behaviour brake is sigmoidal (Hill/logistic) in cumulative *visible*
deaths, bounded below by R_floor:

    brake = 1 / (1 + exp(alpha * ifr * (cum_lagged - cum_half)))
    R_eff = R_floor + (R0 - R_floor) * brake

Two details matter. The driver is looked up in calendar time -- a
per-parent searchsorted on per-zone sorted post-recognition t_infect
arrays -- rather than by generation, which would otherwise mismatch the
timing. And it is lagged by recovery_mean (~16 d) so the brake responds to
deaths that have actually become visible, not to fresh infections; without
the lag the brake bites weeks too early. R_floor keeps R_eff from
collapsing toward zero once cumulative deaths grow.
"""

from __future__ import annotations

import os

import numpy as np

from ..model.model import Params, simulate
from .state import SHARED


def _seed_u_min():
    """The seed-conditioning rule, from `fixed.seed_u_min` in config.yaml.

    `null` (or absent) = the unconditioned draw. `nonzero` = exact zero-truncation of the
    index case's NB draw. A float is a sensitivity-probe mode and is not for fitting.
    SEED_U_MIN in the env overrides, for the gate scripts.
    """
    v = os.environ.get("SEED_U_MIN", SHARED["fx"].get("seed_u_min"))
    if v is None or v == "" or (isinstance(v, str) and v.lower() in ("null", "none", "off")):
        return None
    if isinstance(v, str) and v.lower() == "nonzero":
        return "nonzero"
    u = float(v)
    if not 0.0 <= u < 1.0:
        raise ValueError(f"seed_u_min must be in [0, 1), 'nonzero', or null; got {v!r}")
    return u


def make_params(p, seed_day) -> Params:
    """Build a per-draw `Params` from the merged fixed + calibrated dict `p`."""
    fx = SHARED["fx"]
    return Params(
        R0=p["R0"],
        k=max(p["k"], 0.02),
        gi_shape=SHARED["gi_shape"],
        gi_scale=SHARED["gi_scale"],
        incub_meanlog=SHARED["incub_ml"],
        incub_sdlog=SHARED["incub_sl"],
        jump_prob=p["jump_prob"],
        ifr=p["ifr"],
        # ramp / burial
        ramp_a=fx["ramp_a"],
        w_burial=fx["w_burial"],
        burial_width=fx["burial_width"],
        recovery_mean=fx["recovery_mean"],
        recovery_sd=fx["recovery_sd"],
        # Hill brake. R_floor and hill_n may be CAL or FIXED depending on which reforms this
        # run carries — `p` is merged fixed-then-CAL by every call site, so both lookups are
        # form-agnostic and neither line changes when a reform is switched on.
        R_floor=p["R_floor"],
        cum_half=p["cum_half"],
        hill_n=p["hill_n"],
        brake_lag_days=fx["recovery_mean"],
        # Waning memory. `inf` (the default, and what `fixed.tau_w: .inf` yields) is the
        # plain cumulative driver.
        tau_w=float(p.get("tau_w", fx.get("tau_w", float("inf")))),
        # reporting / seed / recognition
        report_delay_mean=fx["report_delay_mean"],
        response_start_day=SHARED["response_day"],
        seed_n=int(SHARED["seed_n"]),
        seed_day=int(seed_day),
        seed_zone=SHARED["seed_zone"],
        seed_u_min=_seed_u_min(),
        # lab turnaround
        local_turnaround=fx["local_turnaround"],
        unserved_turnaround_early=fx["unserved_turnaround_early"],
        unserved_turnaround_late=fx["unserved_turnaround_late"],
        unserved_switch_day=SHARED["unserved_switch_day"],
        # collection
        collect_delay_machine=fx["collect_delay_machine"],
        collect_delay_nomachine=fx["collect_delay_nomachine"],
        # tri-state PCR
        p_pcr_high=p["ascertainment"],
        p_pcr_mid=0.5 * (p["ascertainment"] + p["ascertainment"] * p["collect_ratio"]),
        p_pcr_low=p["ascertainment"] * p["collect_ratio"],
        aware_threshold=SHARED["aware_threshold"],
        # clinical detection
        p_clin=p["p_clin"],
        clin_beta_a=fx["clin_beta_a"],
        clin_beta_b=fx["clin_beta_b"],
        # isolation. The two tiers are NESTED on one uniform in model.py
        # (`conf_eff = got_pcr & (u["presump"] < p_iso_confirmed)`), so `p_iso_confirmed`
        # is a TOTAL, not an increment, and any draw with p_iso_confirmed < p_iso_presump
        # makes the confirmation channel a strict no-op. Sampling two INDEPENDENT Betas
        # for the two tiers puts ~40% of draws in exactly that state -- an incoherent
        # prior. Sampling `iso_uplift` on [0,1] instead makes the ordering hold by
        # construction and states the mechanism: of the suspects NOT isolated on
        # suspicion, `iso_uplift` are isolated once the PCR result lands.
        p_iso_presump=p["p_iso_presump"],
        p_iso_confirmed=p["p_iso_presump"] + (1.0 - p["p_iso_presump"]) * p["iso_uplift"],
        # cadaver swab
        cadaver_swab_coverage=p["cadaver_swab_coverage"],
        cadaver_swab_sens=fx["cadaver_swab_sens"],
        cadaver_swab_delay_days=fx["cadaver_swab_delay_days"],
        cadaver_swab_scope_all=fx["cadaver_swab_scope_all"],
        # SQ tracing
        sq_trace_eff=p["sq_trace_eff"],
        sq_trace_lag=fx["sq_trace_lag"],
        # contact-LISTING throughput cap. MEASURED, not calibrated — 435 contacts
        # listed/day (trendless vs incidence) / 13.3 contacts per case ~= 33 cases/day
        # ~= 460 per 14 d. Overridable per-replay via p["trace_cap"] so a sweep over
        # {305, 460, 610, inf} needs no config edit. Defaults to inf (no cap).
        trace_cap=float(p.get("trace_cap", fx.get("trace_cap", float("inf")))),
        trace_window=float(fx.get("trace_window", 14.0)),
        # bed-stay caps — FIXED
        bed_stay_suspect_days=fx["bed_stay_suspect_days"],
        bed_stay_post_confirm_days=fx["bed_stay_post_confirm_days"],
        # care-seeking referral + nosocomial R excess. f_refer is FIXED at cfg.fixed (0.05).
        f_refer=fx["f_refer"],
        noso_boost=p["noso_boost"],
        # national awareness brake (shape and driver fixed)
        nat_half=p["nat_half"],
        nat_hill_n=fx["nat_hill_n"],
        nat_brake_driver=fx.get("nat_brake_driver", "deaths"),
        nat_half_days=fx.get("nat_half_days", 0.0),
        nat_floor=p.get("nat_floor", 0.0),  # .get so a config without the key still loads
        # The SLOW national clock, and where the national factor multiplies. All FIXED in
        # config, never sampled; the forward sweep overrides them from its `ramp_arms` record.
        nat_brake_placement=str(fx.get("nat_brake_placement", "gap")),
        nat_scale_T=float(fx.get("nat_scale_T", 0.0)),
        nat_scale_min=float(fx.get("nat_scale_min", 1.0)),
        nat_scale_n=float(fx.get("nat_scale_n", 2.0)),
        # The clock's ONSET, in days past response_start. 88 = the horizon, which makes the
        # clock exactly invisible to the fit; 0 starts it at response start. Never swept.
        nat_scale_t0=float(fx.get("nat_scale_t0", 0.0)),
    )


def zone_r_multiplier(p):
    """(Z,) per-zone multiplier on r_eff.

      m_z = 1 - (1 - rho_access) * s_hetero_z

    `s_hetero` is 0 for zones in provinces as conflict-affected as Ituri and 1 for zones
    with no organised violence, so `rho_access` is the R multiplier for a fully-peaceful,
    better-served province. rho_access = 1 removes the heterogeneity.
    """
    return 1.0 - (1.0 - float(p["rho_access"])) * SHARED["s_hetero"]


def three_tier_mix(alpha_st, beta):
    """Three-tier per-draw effective mobility.
      Tier 1 (short-trips-covered origins, `SHARED['tier1_ituri_idx']` + `SHARED['tier1_nk_idx']`):
        row_norm(α_st · M_flow[i,·] + (1−α_st) · M_st[cohort(i),·])
      Tier 2 (Flowminder-only): M_flow[i,·] unchanged.
      Tier 3 (Flowminder-missing, `SHARED['tier3_idx']`): pure gravity `pop_j / (d[i,j] + d0)^β`.

    `SHARED['M']` is used for tier 1's Flowminder component and tier 2. It has tier-3 rows
    already zeroed (no copy-neighbour fallback) so tier 3 handling is
    entirely gravity-driven.
    """
    M_eff = SHARED["M"].copy()

    # Tier 1 mix (Ituri + NK cohorts): blend rows with their cohort's short-trips profile
    for cohort_key, idx_key in (("st_ituri", "tier1_ituri_idx"), ("st_nk", "tier1_nk_idx")):
        idx = SHARED[idx_key]
        if idx.size == 0:
            continue
        row_flow = M_eff[idx]  # (n_cohort, Z), rows already normed
        row_st = SHARED[cohort_key]  # (Z,) row-normed cohort short-trips row
        mixed = alpha_st * row_flow + (1.0 - alpha_st) * row_st[None, :]
        rs = mixed.sum(1, keepdims=True)
        M_eff[idx] = np.divide(mixed, rs, out=np.zeros_like(mixed), where=rs > 0)

    # Tier 3 (Flowminder-missing zones): pure gravity row per source zone
    t3 = SHARED["tier3_idx"]
    if t3.size > 0:
        # pop_j / (d + d0)^β for rows in tier 3, all destinations j
        d_sub = SHARED["gravity_dist"][t3]  # (n3, Z)
        pop_row = SHARED["gravity_num_pop"][t3]  # (n3, Z) — same as pop_j broadcast
        Mg = pop_row / np.power(d_sub + SHARED["gravity_d0"], beta)
        # zero self-loops on the sub-block
        for k, i in enumerate(t3):
            Mg[k, i] = 0.0
        rs = Mg.sum(1, keepdims=True)
        M_eff[t3] = np.divide(Mg, rs, out=np.zeros_like(Mg), where=rs > 0)

    return M_eff


def simulate_draw(p, seed_day, sim_seed):
    # cap_vals[1] (the PHEIC->Bunia window) is CALIBRATED per draw as p["cap_phei"].
    cap_days, cap_vals = SHARED["cap"]
    cap_vals = cap_vals.copy()
    cap_vals[1] = float(p["cap_phei"])
    cap_scaled = (cap_days, cap_vals * SHARED["fx"]["cap_multiplier"])
    # gravity_beta is FIXED, not calibrated per draw
    M_eff = three_tier_mix(float(p["short_trips_alpha"]), float(SHARED["fx"]["gravity_beta"]))
    return simulate(
        make_params(p, seed_day),
        SHARED["zones"],
        M_eff,
        SHARED["served"],
        SHARED["cutoff_day"],
        np.random.default_rng(int(sim_seed)),
        collect_high_from_day=SHARED["collect_high"],
        confirm_capacity=cap_scaled,
        queue_query_days=SHARED["backlog_query_days"],
        care_destination=SHARED["care_destination"],  # referral map
        zone_r_mult=zone_r_multiplier(p),  # conflict/access heterogeneity
        listing_load=SHARED.get("listing_load"),
    )  # exogenous tracing load


def cumulative_by_day(times, day_ints):
    if times.size == 0:
        return np.zeros(len(day_ints))
    t = np.sort(times)
    return np.searchsorted(t, day_ints, side="right").astype(float)


def _new_in_window(times, win_edges):
    """Count events in each (left, right] window (np.diff of cumulative)."""
    if times.size == 0:
        return np.zeros(len(win_edges) - 1)
    c = cumulative_by_day(times, np.asarray(win_edges))
    return np.diff(c)


def confirmed_iso_census(inf, day_ints):
    """Model CONFIRMED-AND-ISOLATED stock on each query day: cases that are isolated, in the iso
    window, AND PCR-confirmed by that day. Clean tracing comparator (true cases <-> true cases,
    no positivity scaling) against the SitRep-validated daily confirmed-in-iso series."""
    iso = inf["isolated"].to_numpy()
    s = inf["iso_start"].to_numpy()
    e = inf["iso_end"].to_numpy()
    det = inf["detected"].to_numpy()
    td = inf["t_detect"].to_numpy()
    m = iso & np.isfinite(s) & np.isfinite(e) & det & np.isfinite(td)
    s = s[m]
    e = e[m]
    td = td[m]
    return np.array([int(((s <= d) & (e >= d) & (td <= d)).sum()) for d in day_ints], float)


def suspect_iso_census(inf, day_ints):
    """Model UNCONFIRMED-AND-ISOLATED stock per day (isolated, in window, NOT YET confirmed).
    Compared to the observed SUSPECT-in-iso census via /suspect_positivity (PPC overlay only)."""
    iso = inf["isolated"].to_numpy()
    s = inf["iso_start"].to_numpy()
    e = inf["iso_end"].to_numpy()
    det = inf["detected"].to_numpy()
    td = inf["t_detect"].to_numpy()
    m = iso & np.isfinite(s) & np.isfinite(e)
    s = s[m]
    e = e[m]
    det = det[m]
    td = td[m]
    out = np.zeros(len(day_ints), float)
    for i, d in enumerate(day_ints):
        in_bed = (s <= d) & (e >= d)
        confirmed_now = in_bed & det & (td <= d)
        out[i] = (in_bed & ~confirmed_now).sum()
    return out


def discrete_gi(gi_shape, gi_scale, nmax=60):
    """Discretised generation-interval pmf."""
    from scipy.stats import gamma as _gamma

    cdf = _gamma.cdf(np.arange(nmax + 1), gi_shape, scale=gi_scale)
    g = np.diff(cdf)
    g[0] = 0.0
    return g / g.sum()


def _renewal_rt(inc, g, smooth=7):
    """Renewal-equation effective Rt from a daily incidence series."""
    s = np.convolve(inc, np.ones(smooth) / smooth, mode="same")
    rt = np.full(len(s), np.nan)
    for tt in range(1, len(s)):
        k = np.arange(1, min(len(g) - 1, tt) + 1)
        d = float(np.sum(g[k] * s[tt - k]))
        if d > 1e-6:
            rt[tt] = s[tt] / d
    return rt


def _latest_rt(t_infect):
    """Model effective Rt averaged over the scoring window, for the epiforecasts
    latest-R anchor. Uses the SAME renewal estimator as the 07_rt figure so the scored
    target and the plotted curve agree by construction.

    The window ends `rt_lag` days before the cutoff: a 7-day centred smooth biases the
    final ~3 days, so the last few days of the curve are not a fair comparator.
    """
    cut = int(SHARED["cutoff_day"])
    full = np.arange(0, cut + 1)
    ti = np.asarray(t_infect, float)
    ti = ti[np.isfinite(ti) & (ti <= cut) & (ti >= 0)]
    if ti.size == 0:
        return np.nan
    inc = np.bincount(np.floor(ti).astype(int), minlength=len(full))[: len(full)].astype(float)
    rt = _renewal_rt(inc, SHARED["gi_disc"])
    hi = cut - int(SHARED["rt_lag"])
    lo = hi - int(SHARED["rt_window"]) + 1
    seg = rt[max(lo, 0) : hi + 1]
    seg = seg[np.isfinite(seg)]
    return float(np.mean(seg)) if seg.size else np.nan


def expected_summaries(inf) -> dict:
    """Per-draw expected summary stats. Per-zone spatial stats use care_zone (referred
    cases attributed to their CTE) rather than zone_idx (the infection zone)."""
    det = inf["detected"].to_numpy()
    td = inf["t_detect"].to_numpy()
    zi_care = inf["care_zone"].to_numpy()  # care_zone, not zone_idx, for spatial obs
    m = det & np.isfinite(td)
    # confirmed: 3-day-new on the case-curve summary edges; daily cumulative for PPC
    nat_cum_daily = cumulative_by_day(td[m], SHARED["day_ints"])
    conf_new = _new_in_window(td[m], SHARED["conf_edges"])
    # zone-final cumulative confirmed — attributed to care_zone
    mc = m & (td <= SHARED["cutoff_day"])
    zone_final = np.bincount(zi_care[mc], minlength=SHARED["Z"]).astype(float)
    # deaths: binned rail — incident deaths per (two-weekly + terminal) cell.
    died = inf["died"].to_numpy()
    tdh = inf["t_death"].to_numpy()
    cd = m & died  # full-length boolean mask
    # death_day computed for ALL cases (inf where td inf); safe to subset either way
    death_day_all = np.where(cd, np.maximum(td, tdh), np.inf)
    deaths_binned = _new_in_window(death_day_all[cd], SHARED["death_edges"])
    # per-zone cumulative confirmed DEATHS at horizon. Dropped from
    # scoring (DM-deaths unreliable spatially); still computed for PPC/diagnosis figs.
    md = cd & (death_day_all <= SHARED["cutoff_day"])
    zone_deaths_final = np.bincount(zi_care[md], minlength=SHARED["Z"]).astype(float)
    # confirmed-in-iso
    confirmed_iso = confirmed_iso_census(inf, SHARED["isoval_day_ints"])
    # model BDBV-suspect-in-iso census (PPC-overlay-only before; now scored
    # to break an otherwise bimodal p_iso_presump posterior).
    suspect_iso = suspect_iso_census(inf, SHARED["isoval_day_ints"])
    # daily new ADMISSIONS to isolation (model TRUE admissions; observed scaled by positivity).
    # New admission = case enters iso (iso_start in day window). Bin to adm_edges (3-day).
    iso = inf["isolated"].to_numpy()
    iso_start = inf["iso_start"].to_numpy()
    valid_adm = iso & np.isfinite(iso_start)
    adm_new = _new_in_window(iso_start[valid_adm], SHARED["adm_edges"])
    # daily new RECOVERED (confirmed survivors discharged from iso).
    # iso_end = abs discharge day for cases that were isolated. Recovered = detected & not died.
    iso_end = inf["iso_end"].to_numpy()
    valid_rec = iso & det & (~died) & np.isfinite(iso_end) & np.isfinite(td)
    rec_new = _new_in_window(iso_end[valid_rec], SHARED["rec_edges"])
    rt_latest = _latest_rt(inf["t_infect"].to_numpy())  # epiforecasts latest-R anchor
    return {
        "rt_latest": rt_latest,
        "nat_cum_daily": nat_cum_daily,
        "conf_new": conf_new,
        "zone_final": zone_final,
        "zone_deaths_final": zone_deaths_final,  # figs only (unscored)
        "deaths_binned": deaths_binned,
        "confirmed_iso": confirmed_iso,
        "suspect_iso": suspect_iso,
        "adm_new": adm_new,
        "rec_new": rec_new,
    }


def dm_lump(v: np.ndarray, top_idx: np.ndarray, other_mask: np.ndarray) -> np.ndarray:
    """Aggregate a length-Z vector to (top_n + 1) categories: the top_n zones
    (in `top_idx` order) and one 'other' bucket = sum over the remaining
    `other_mask=True` positions. Spatial DM lumping."""
    return np.concatenate([v[top_idx], [v[other_mask].sum()]])
