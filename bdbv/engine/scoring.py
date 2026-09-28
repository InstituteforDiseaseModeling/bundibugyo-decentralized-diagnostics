"""Likelihood of one simulated trajectory against the observed data, the SIR/PPC worker
tasks built on it, and pseudo-marginal seed selection from a stored posterior."""

from __future__ import annotations

import numpy as np

from ..model.likelihood import (
    beta_binomial_loglik,
    composite_loglik,
    dirichlet_multinomial_loglik,
    nb_loglik,
    nb_lower_bound_loglik,
    normal_loglik,
)
from ..model.priors import sample_prior
from .simulation import (
    confirmed_iso_census,
    cumulative_by_day,
    dm_lump,
    expected_summaries,
    simulate_draw,
    suspect_iso_census,
)
from .state import CAL, FIXED_FROM_CAL, SHARED, theta_out_of_support


def select_within_theta(logL, rng):
    """Per-theta (per-row) likelihood-proportional selection among the M
    seeds. Returns an index array (N,) into the M columns. Weights are normalised
    WITHIN each theta (w_k = L_k / sum_j L_j) — never globally, which would
    double-count the likelihood and bias the parameter posterior. Burnout columns
    (logL ~ -inf) get weight ~0 by construction."""
    logL = np.asarray(logL, dtype=float)
    N, M = logL.shape
    sel = np.zeros(N, dtype=int)
    for i in range(N):
        row = logL[i]
        mx = row.max()
        if not np.isfinite(mx):
            sel[i] = 0
            continue
        w = np.exp(row - mx)
        s = w.sum()
        sel[i] = (
            int(np.argmax(row)) if (not np.isfinite(s) or s <= 0) else int(rng.choice(M, p=w / s))
        )
    return sel


def draw_trajectories(post, n, seed=0):
    """Draw n single-seed joint-posterior samples (theta, seed_day,
    sim_seed) from a pseudo-marginal posterior.npz. Each stored theta carries mass
    1 (theta's are already posterior-distributed); the seed within a theta is
    picked likelihood-proportionally. Sampling is WITH replacement, so theta- and
    (theta,seed)-duplicates are expected and correct."""
    rng = np.random.default_rng(seed)
    pm_seeds, pm_logL = post["pm_seeds"], post["pm_logL"]
    N = pm_logL.shape[0]
    rows = rng.integers(0, N, size=n)
    sel = select_within_theta(pm_logL[rows], rng)
    thetas = {k: post[k][rows] for k in CAL}
    return thetas, post["seed_day"][rows], pm_seeds[rows, sel]


def weighted_band(post):
    """Full-latent PPC band without collapsing to one seed — returns the
    M seeds (N,M) and their within-theta weights (N,M)."""
    pm_logL = post["pm_logL"]
    mx = pm_logL.max(axis=1, keepdims=True)
    w = np.where(np.isfinite(pm_logL), np.exp(pm_logL - mx), 0.0)
    w /= w.sum(axis=1, keepdims=True)
    return post["pm_seeds"], w


def log_likelihood(p, seed_day, sim_seed) -> float:
    inf = simulate_draw(p, seed_day, sim_seed)
    e = expected_summaries(inf)
    queue_at = inf.attrs.get("queue_at_query", np.zeros(SHARED["backlog_query_days"].size))
    n_sb = SHARED["n_scored_backlog"]

    # adm_new, rec_new, suspect_iso REMOVED from scoring. All three derive
    # from the same soft SR isolation log where suspect->confirmed->recovered flows
    # don't reconcile; scoring them was manufacturing systematic bias into the joint
    # likelihood. suspect_positivity is now a cosmetic overlay constant only (see
    # _expected_ppc). No division by suspect_positivity happens in the scored path.
    # death rail = leading two-weekly NB cells + one terminal lower-bound
    # cell. n_nb = number of leading NB cells; the remainder ([n_nb:], length 1) is
    # the soft-lower-bound terminal cell over the confirmation-batch surge window.
    n_nb = int(SHARED["n_death_nb_cells"])
    obs_d = SHARED["obs_deaths_binned"]
    mod_d = e["deaths_binned"]
    base_ll = composite_loglik(
        [
            (SHARED["obs_conf_new"], e["conf_new"]),
            (obs_d[:n_nb], mod_d[:n_nb]),  # two-weekly NB death cells
            (SHARED["obs_backlog_scaled"], queue_at[:n_sb]),
            (SHARED["obs_confirmed_iso"], e["confirmed_iso"]),
        ],
        SHARED["disp"],
    )
    # Terminal death cell. Under "lower_bound" it is a soft lower bound (penalise only
    # where the model under-predicts an under-counted target), which leaves over-prediction
    # free and pushes latent burden UP. "two_sided" is NEUTRAL: a symmetric NB cell against
    # the observed terminal count (optionally inflated by `terminal_completeness` for
    # report-lag incompleteness). No-op when n_nb == n_cells.
    if SHARED["death_terminal_mode"] == "two_sided":
        base_ll += nb_loglik(
            SHARED["obs_deaths_terminal"], mod_d[n_nb:], SHARED["death_terminal_disp"]
        )
    else:
        base_ll += nb_lower_bound_loglik(obs_d[n_nb:], mod_d[n_nb:], SHARED["disp"])

    # effective-Rt anchor. epiforecasts' LATEST reproduction number is the
    # only published quantity that constrains the BRAKED state directly, which is what
    # `nat_half` controls — the count cells constrain cumulative size, not the current
    # rate of decline. Scored as a Normal against their credible band, using the same
    # renewal estimator as the 07_rt figure. Soft by construction (one cell against ~50),
    # so it informs rather than dominates.
    if SHARED["rt_target_sd"] > 0 and np.isfinite(e["rt_latest"]):
        base_ll += normal_loglik(
            [SHARED["rt_target_mean"]], [e["rt_latest"]], SHARED["rt_target_sd"]
        )

    # SPLIT the spatial case-share likelihood into two independent terms.
    # ~95% of the spatial misfit sits in the DM component: the
    # model over-disperses cases out of the concentrated core into the diffuse
    # "other" tail (obs ~0.06 share vs PPC ~0.17). A single 16-cell DM weights
    # "which core zone leads" and "did it jump to the rest of DRC" equally, but
    # only the second breaks the frontier forecast. So:
    #   (a) CORE-vs-PERIPHERY (forecast-critical): Beta-Binomial on the observed
    #       "other" count vs the model's predicted "other" share. Concentration
    #       `betabin_s` alone sets tolerance (s=300 => ~2x overshoot costs ~6.3
    #       log-units). This is where the leakage-out-of-core gets pinned.
    #   (b) WITHIN-CORE (we care far less): Dirichlet-multinomial on the TOP-15
    #       zones ONLY ("other" dropped), alpha=dm_alpha_cases. Scores how core
    #       cases distribute among the 15 named zones, orthogonal to the split.
    # the SPLIT's p_other_model uses `split_background`, NOT `zone_background`. The obs
    # side carries no background, so only 0.0 makes the scored share symmetric raw-vs-raw.
    mod_lumped = dm_lump(
        e["zone_final"] + SHARED["split_background"], SHARED["dm_top_idx"], SHARED["dm_other_mask"]
    )
    obs_lumped = SHARED["obs_zone_final_lumped"]
    mod_total = float(mod_lumped.sum())
    p_other_model = (float(mod_lumped[-1]) / mod_total) if mod_total > 0 else 0.0
    split_ll = beta_binomial_loglik(
        float(obs_lumped[-1]), float(obs_lumped.sum()), p_other_model, SHARED["betabin_s"]
    )
    # Both spatial terms are SCORED, and kept SEPARATE: (a) split_ll = Beta-Binomial on
    # the forecast-critical "frac other" share; (b) core_ll = DM on the
    # TOP-15 zones ONLY ("other" dropped), so it constrains the within-core distribution
    # orthogonally to the split rather than being re-lumped into one 16-cell DM. The
    # ledger's open lever: the split alone (at s=300) was 10x tighter than the DM and
    # forced jump_prob down / R0 up; re-adding the DM + loosening the split rebalances
    # the two spatial constraints.
    core_ll = dirichlet_multinomial_loglik(
        obs_lumped[:-1], mod_lumped[:-1], SHARED["dm_alpha_cases"]
    )
    base_ll += split_ll + core_ll  # both spatial terms scored

    # epi-joint soft cell at horizon
    if SHARED.get("epi_joint_enabled"):
        ti = inf["t_infect"].to_numpy()
        lat_at_horizon = int((ti <= SHARED["cutoff_day"]).sum())
        epi_ll = nb_loglik(
            np.array([SHARED["epi_estimate"]]),
            np.array([max(lat_at_horizon, 1)]),
            SHARED["epi_disp"],
        )
        return base_ll + float(epi_ll)
    return base_ll


def score_draw(task):
    i, pseed, sseed = task
    rng = np.random.default_rng(pseed)
    p = {k: float(SHARED["fx"][k]) for k in FIXED_FROM_CAL}  # fixed values first,
    p.update(
        {k: sample_prior(SHARED["priors"][k], rng) for k in CAL}
    )  # then sampled (CAL wins on overlap; none expected)
    # seed_day comes from the calibrated `seed_day_offset` param.
    seed_day = round(SHARED["start_mean"] + p["seed_day_offset"])
    return (i, log_likelihood(p, seed_day, sseed), tuple(p[k] for k in CAL), seed_day)


def ppc_draw(task):
    vals, seed_day, sseed = task
    if theta_out_of_support(
        vals
    ):  # mirror _emcee_log_prob — skip zero-mass draws (see theta_out_of_support)
        return None
    p = {k: float(SHARED["fx"][k]) for k in FIXED_FROM_CAL}  # see score_draw
    p.update(dict(zip(CAL, vals)))
    inf = simulate_draw(p, seed_day, sseed)
    e = expected_summaries(inf)
    ti = inf["t_infect"].to_numpy()
    lat_cum = cumulative_by_day(ti[ti <= SHARED["cutoff_day"]], SHARED["day_ints"])
    queue_at = inf.attrs.get("queue_at_query", np.zeros(SHARED["backlog_query_days"].size))
    confirmed_iso_full = confirmed_iso_census(inf, SHARED["isoval_day_ints"])
    suspect_iso = suspect_iso_census(inf, SHARED["isoval_day_ints"])
    iso = inf["isolated"].to_numpy()
    s = inf["iso_start"].to_numpy()
    end = inf["iso_end"].to_numpy()
    mok = iso & np.isfinite(s) & np.isfinite(end)
    s = s[mok]
    end = end[mok]
    iso_stock = np.array([int(((s <= d) & (end >= d)).sum()) for d in SHARED["day_ints"]], float)
    # suspect_positivity is FIXED, not calibrated per draw
    adm_pp = e["adm_new"] / SHARED["fx"]["suspect_positivity"]
    return (
        e["nat_cum_daily"],
        lat_cum,
        e["conf_new"],
        confirmed_iso_full,
        suspect_iso,
        iso_stock,
        queue_at,
        e["zone_final"],
        e["deaths_binned"],
        adm_pp,
        e["rec_new"],
        e["zone_deaths_final"],
    )


def ess_of(logL):
    w = np.exp(logL - np.nanmax(logL))
    w /= w.sum()
    return 1.0 / np.sum(w**2), w
