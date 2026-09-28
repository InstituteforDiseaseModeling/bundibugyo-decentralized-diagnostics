"""Intervention layer: footprint, placement, scenarios.

Deliberately thin. All parameter construction, mobility, and fixed values come from
the shared calibration engine so forward replay is identical to the fitted path. This
module only adds what the calibration never had: where machines go, what that does to a
zone's detection, and how samples ship.
"""

from __future__ import annotations

import numpy as np

from bdbv import engine as E

# ---------------------------------------------------------------------------
# Travel-time / shipping
# ---------------------------------------------------------------------------


def ship_days(tt_minutes: np.ndarray, cfg_iv: dict) -> np.ndarray:
    """(Z,Z) one-way shipping time in days from the OSRM travel-time matrix."""
    return tt_minutes / float(cfg_iv["ship_minutes_per_day"])


def ship_to_nearest(chosen: list[int], ship_full: np.ndarray) -> np.ndarray:
    """(Z,) days to the nearest chosen machine zone (+inf if unreachable)."""
    Z = ship_full.shape[0]
    if not chosen:
        return np.full(Z, np.inf)
    sub = ship_full[:, list(chosen)]
    fin = np.isfinite(sub)
    return np.where(fin.any(1), np.nanmin(np.where(fin, sub, np.inf), 1), np.inf)


def ship_turnaround(chosen: list[int], ship_full: np.ndarray, cfg_iv: dict) -> np.ndarray:
    """(Z,) effective turnaround if a zone ships its samples to the nearest machine."""
    d = ship_to_nearest(chosen, ship_full)
    return (
        float(cfg_iv["tau_process_days"])
        + float(cfg_iv["ship_fixed_days"])
        + float(cfg_iv["friction"]) * d
    )


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def random_placement(budget: int, eligible: np.ndarray, rng) -> list[int]:
    """Uniform sample of eligible (currently unserved) zones — the baseline that
    'smart' placement is measured against."""
    elig = np.where(eligible)[0]
    if elig.size == 0:
        return []
    return [int(z) for z in rng.choice(elig, size=min(budget, elig.size), replace=False)]


def greedy_order(
    pre_trees, pre_params, ctx, cfg_iv, deploy_day, n_cand=150, kmax=200, log=print
) -> list[int]:
    """Marginal-averted submodular greedy over an ensemble of pre-pass SQ trees.

    Returns a single ORDER; every budget is a prefix of it (so 25/50/100/200 all come
    from one pass).
    """
    Z = ctx["Z"]
    eligible = ~ctx["served_mask"]
    # shortlist by ensemble burden so the greedy is affordable
    burden = np.zeros(Z)
    for T in pre_trees:
        if T.get("n", 0):
            burden += np.bincount(T["zone_idx"], minlength=Z)
    cand = [int(z) for z in np.argsort(burden)[::-1] if eligible[z] and burden[z] > 0][:n_cand]
    log(
        f"[greedy] {len(cand)} candidates from {len(pre_trees)} pre-pass trees; "
        f"top-5 burden zones {[ctx['zones'][z] for z in np.argsort(burden)[::-1][:5]]}"
    )

    # ---- lazy greedy (CELF). Averted burden is submodular in the placement set, so a
    # stale marginal gain is an UPPER BOUND on the current one. Pop the best bound,
    # refresh only that candidate, and accept it if it still tops the next bound. This
    # returns the IDENTICAL set to full greedy at a fraction of the evaluations
    # (~750 vs ~30,000 here), which is what makes an honest marginal-averted picker
    # affordable.
    import heapq

    def total_averted(sel):
        return float(
            sum(
                _averted(T, p, ctx, cfg_iv, sel, deploy_day) for T, p in zip(pre_trees, pre_params)
            )
        )

    n_eval = 0
    chosen: list[int] = []
    base_tot = 0.0
    heap: list[tuple[float, int]] = []
    for z in cand:
        g = total_averted([z]) - base_tot
        n_eval += 1
        heap.append((-g, z))
    heapq.heapify(heap)

    while len(chosen) < min(kmax, len(cand)) and heap:
        stale = 0
        while True:
            _, z = heapq.heappop(heap)
            g = total_averted(chosen + [z]) - base_tot
            n_eval += 1
            if not heap or g >= -heap[0][0] - 1e-9:
                break
            heapq.heappush(heap, (-g, z))
            stale += 1
            if stale > len(cand):  # pathological; take what we have
                break
        if g <= 0:
            log(f"[greedy] stopping at {len(chosen)}: no positive marginal gain")
            break
        chosen.append(int(z))
        base_tot += g
        if len(chosen) % 25 == 0:
            log(f"[greedy]   {len(chosen):3d} placed, last gain {g:.1f}, evals {n_eval:,}")
    log(
        f"[greedy] done: {len(chosen)} zones, {n_eval:,} marginal evaluations "
        f"(full greedy would need ~{min(kmax, len(cand)) * len(cand):,})"
    )
    return chosen


def _averted(T, params, ctx, cfg_iv, chosen, deploy_day) -> float:
    """Infections removed by placing machines at `chosen` (LOCAL mode, no SDB change)."""
    from bdbv.model.tree import prune

    if not chosen or T.get("n", 0) == 0:
        return 0.0
    mask = np.zeros(ctx["Z"], bool)
    mask[list(chosen)] = True
    sc = build_scenario(
        mask, ctx, cfg_iv, mode="local", sdb_scope="none", sdb_target=None, params=params
    )
    return float(prune(T, params, sc, deploy_day)["removed"].sum())


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def build_scenario(
    cs_mask: np.ndarray, ctx: dict, cfg_iv: dict, *, mode: str, sdb_scope: str, sdb_target, params
) -> dict:
    """Per-zone counterfactual detection state for one intervention arm.

    mode='local'      machines serve only their own zone.
    mode='transport'  unserved zones may ship samples to the nearest machine, which
                      lifts them out of the slow unserved turnaround.

    The tri-state PCR ladder is reconstructed from the DRAW's own parameters
    (ascertainment x collect_ratio), matching `bdbv.engine.make_params`, so a machine
    zone gets that draw's `p_pcr_high` rather than a hardcoded constant.
    """
    Z = ctx["Z"]
    cs_mask = np.asarray(cs_mask, bool)
    p_high = float(params.p_pcr_high)
    p_mid = float(params.p_pcr_mid)

    onsite = cs_mask.copy()  # machine physically in the zone
    turn = np.full(Z, np.inf)
    turn[cs_mask] = float(cfg_iv["turnaround_machine"])
    local = cs_mask.copy()
    pcr = np.where(cs_mask, p_high, 0.0)
    applies = cs_mask.copy()

    if mode == "transport":
        st = ship_turnaround([int(z) for z in np.where(cs_mask)[0]], ctx["ship_full"], cfg_iv)
        # only UNSERVED zones are lifted by shipping — a zone already inside a 180-min
        # catchment of an online lab is the SQ, not something transport improves
        reach = np.isfinite(st) & ~cs_mask & ~ctx["served_mask"]
        # shipping lifts a zone to the 'aware' PCR tier and the shipped turnaround,
        # but does NOT put a machine in it (collection delay stays the non-machine one)
        turn = np.where(reach, st, turn)
        pcr = np.where(reach, np.maximum(pcr, p_mid), pcr)
        local = local | reach
        applies = applies | reach
    elif mode != "local":
        raise ValueError(f"unknown mode {mode!r}")

    # --- safe-and-dignified burial scope
    if sdb_scope == "none" or sdb_target is None:
        sdb_zones = np.zeros(Z, bool)
        sdb_cov = float(params.cadaver_swab_coverage)
    elif sdb_scope == "machine":
        sdb_zones = cs_mask.copy()
        sdb_cov = float(sdb_target)
    elif sdb_scope == "all":
        sdb_zones = np.ones(Z, bool)
        sdb_cov = float(sdb_target)
    else:
        raise ValueError(f"unknown sdb_scope {sdb_scope!r}")
    # "raise coverage to at least X" — never lower a draw whose SQ value already exceeds it
    sdb_cov = max(sdb_cov, float(params.cadaver_swab_coverage))
    # zones only 'apply' for the detection splice; SDB is gated separately in prune()
    return {
        "applies": applies,
        "pcr_prob": pcr,
        "turnaround": turn,
        "onsite": onsite,
        "local": local,
        "sdb_zones": sdb_zones,
        "sdb_coverage": sdb_cov,
    }


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def build_ctx(cfg: dict, shared: dict) -> dict:
    """Augment the shared engine context with the intervention-side geography."""
    iv = cfg["intervention"]
    zones = list(shared["zones"])
    Z = len(zones)
    from bdbv.model import data as D

    tt = D.load_travel_time_matrix().reindex(index=zones, columns=zones).to_numpy(dtype=float)
    served = np.asarray(shared["served"], dtype=float)
    ctx = {
        "zones": zones,
        "Z": Z,
        "served": served,
        "served_mask": np.isfinite(served) & (served <= float(shared["cutoff_day"])),
        "ship_full": ship_days(tt, iv),
        "cutoff_day": int(shared["cutoff_day"]),
        "horizon_day": int(shared["cutoff_day"]) + int(iv["horizon_days"]),
        "marks": [int(shared["cutoff_day"]) + int(m) for m in iv["report_marks"]],
    }
    return ctx


def make_params(post: dict, i: int, shared: dict):
    """One draw -> Params, via the shared engine so it matches the calibration path."""
    p = {k: float(shared["fx"][k]) for k in E.FIXED_FROM_CAL}
    p.update({k: float(post[k][i]) for k in E.CAL})
    return E.make_params(p, int(post["seed_day"][i])), p


def _sim_kw(p, shared):
    cap_days, cap_vals = shared["cap"]
    cv = cap_vals.copy()
    cv[1] = float(p["cap_phei"])
    return {
        "max_infections": int(shared.get("_max_inf", 1_500_000)),
        "collect_high_from_day": shared["collect_high"],
        "confirm_capacity": (cap_days, cv * shared["fx"]["cap_multiplier"]),
        "queue_query_days": shared["backlog_query_days"],
        "care_destination": shared["care_destination"],
        # Conflict/access heterogeneity. Computed by the shared engine's
        # own helper (1 - (1 - rho_access) * s_hetero) so it is identical to what the
        # calibration used -- never reimplemented here.
        "zone_r_mult": E.zone_r_multiplier(p),
    }


def build_history(post: dict, i: int, shared: dict, cutoff_day: int):
    """Phase 1 — the data-conditioned history, run to the cutoff with the STORED seed.

    This is bit-identical to the calibration path, so every
    forward continuation is anchored to a trajectory the likelihood actually scored.
    Returns (hist_tree, params, p_dict); `hist_tree['state']` carries the frontier.
    """
    from bdbv.model.tree import simulate_tree

    prm, p = make_params(post, i, shared)
    M_eff = E.three_tier_mix(float(p["short_trips_alpha"]), float(shared["fx"]["gravity_beta"]))
    T = simulate_tree(
        prm,
        shared["zones"],
        M_eff,
        shared["served"],
        int(cutoff_day),
        np.random.default_rng(int(post["sim_seed"][i])),
        resume=None,
        collect_frontier=True,
        **_sim_kw(p, shared),
    )
    return T, prm, p


def continue_tree(hist: dict, prm, p, shared: dict, horizon_day: int, rng_seed: int):
    """Phase 2 — resume from the cutoff frontier with a fresh stream, then join.

    Everything at or before the cutoff is IDENTICAL across continuations; only the
    future diverges. `rng_seed` selects the branch (continuation 0 uses the stored
    seed so it remains the canonical realisation)."""
    from bdbv.model.tree import join_trees, simulate_tree

    if hist.get("n", 0) == 0 or hist["state"]["frontier"]["zone"].size == 0:
        return hist
    M_eff = E.three_tier_mix(float(p["short_trips_alpha"]), float(shared["fx"]["gravity_beta"]))
    fut = simulate_tree(
        prm,
        shared["zones"],
        M_eff,
        shared["served"],
        int(horizon_day),
        np.random.default_rng(int(rng_seed)),
        resume=hist["state"],
        collect_frontier=False,
        **_sim_kw(p, shared),
    )
    return join_trees(hist, fut)


def build_tree(
    post: dict,
    i: int,
    shared: dict,
    horizon_day: int,
    sim_seed: int | None = None,
    cutoff_day: int | None = None,
    k: int = 0,
):
    """Convenience: full two-phase forward tree for draw i, continuation k."""
    cut = int(cutoff_day if cutoff_day is not None else shared["cutoff_day"])
    hist, prm, p = build_history(post, i, shared, cut)
    seed = int(post["sim_seed"][i]) if sim_seed is None else int(sim_seed)
    if k:
        seed = int(seed) ^ (0x9E3779B97F4A7C15 * (k + 1) & ((1 << 63) - 1))
    return continue_tree(hist, prm, p, shared, horizon_day, seed), prm, p
