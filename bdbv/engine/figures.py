"""Diagnostic figures for a calibration run: posterior-predictive bands, the scored
targets, likelihood diagnosis, spatial fit, capacity/isolation, and the posterior."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..model.likelihood import beta_binomial_loglik
from .simulation import dm_lump
from .state import APPLIED_REFORMS, CAL, FIG, SHARED


def _band(ax, x, arr, color, label, alpha=0.3):
    """Plot a 5-50-95% band with a label."""
    q = np.percentile(arr, [5, 50, 95], axis=0)
    ax.fill_between(x, q[0], q[2], alpha=alpha, color=color)
    ax.plot(x, q[1], color=color, lw=2, label=label)


def _brake_title() -> str:
    """One-line description of THIS run's brake configuration, for figure titles.

    Derived from the live config rather than written as a literal, so a figure can never
    assert a brake design the run did not actually use.
    """
    fx = SHARED.get("fx", {})
    t0 = float(fx.get("nat_scale_t0", 0.0))
    nat = (
        f"national clock onset +{t0:.0f} d = at/after the horizon, so S_nat = 1 in-window"
        if t0 > 0
        else f"national clock from response start (t0 = {t0:.0f})"
    )
    reforms = ", ".join(APPLIED_REFORMS) if APPLIED_REFORMS else "none"
    return f"{nat}; reforms open: {reforms}; betabin_s={fx.get('betabin_s', float('nan')):.0f}"


def fig_all_in_one(
    ctx,
    nat_pp,
    lat_pp,
    conf_new_pp,
    deaths_binned_pp,
    zone_final_pp,
    queue_pp,
    confirmed_iso_pp,
    suspect_iso_pp,
    iso_stock_pp,
    adm_pp,
    rec_pp,
    ess,
    N,
    prior_arrays=None,
) -> None:
    """The headline dashboard — every observable the model is being scored against,
    in a 4x3 grid (12 panels), including admissions + recovered flow panels AND
    prior-PPC overlay (light grey) alongside posterior-PPC (color)."""
    dates = ctx["dates"]
    conf_edges = ctx["conf_edges"]
    conf_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(conf_edges[i + 1]))
        for i in range(len(conf_edges) - 1)
    ]
    death_edges = ctx["shared"]["death_edges"]
    death_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(death_edges[i + 1]))
        for i in range(len(death_edges) - 1)
    ]
    rec_edges = ctx["rec_edges"]
    rec_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rec_edges[i + 1]))
        for i in range(len(rec_edges) - 1)
    ]
    isoval = ctx["isoval"]
    iso_dates = list(isoval["date"])
    n_scored_bk = ctx["shared"]["n_scored_backlog"]
    bk_dates_scored = [pd.Timestamp(d) for d in ["2026-05-25", "2026-06-01"][:n_scored_bk]]
    bk_dates_overlay = [pd.Timestamp(d) for d in ctx["shared"]["overlay_backlog_dates"]]
    bk_dates = bk_dates_scored + bk_dates_overlay  # full sequence aligned with queue_pp columns
    pos = ctx["shared"]["suspect_positivity"]

    # 3x4 LANDSCAPE (PPT-friendly). Scored streams fill rows 1-2, overlays row 3.
    # Row 1 (SCORED): cum-confirmed | 3-day-new confirmed | deaths | confirmed-in-iso
    # Row 2 (SCORED): frac-other split | within-core DM | latent-burden (anchor) | Bunia backlog
    # Row 3 (OVERLAY + text): suspect-in-iso | full-window iso stock | 3-day-new recovered | text
    # (3-day-new admissions DROPPED — weakest overlay.)
    fig, ax = plt.subplots(3, 4, figsize=(28, 16))
    ax = ax.ravel()
    pa = prior_arrays or {}
    # (1) cum confirmed national
    if "nat" in pa:
        _band(ax[0], dates, nat_pp, "#2171b5", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[0], dates, nat_pp, "#2171b5", "Posterior PPC")
    ax[0].plot(dates, ctx["obs_nat_cum"], "k.-", label="Observed")
    ax[0].set_title(f"[derived from 3-day new] Cum confirmed daily — ESS={ess:.0f}/{N}")
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", rotation=45)
    # (2) 3-day-new confirmed (the temporal identifier)
    if "conf" in pa:
        _band(ax[1], conf_mid, conf_new_pp, "#d95f0e", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[1], conf_mid, conf_new_pp, "#d95f0e", "Posterior PPC")
    ax[1].plot(conf_mid, ctx["shared"]["obs_conf_new"], "k.-", label="Observed")
    ax[1].set_title("[SCORED] 3-day-new confirmed (temporal ID)")
    ax[1].legend(fontsize=8)
    ax[1].tick_params(axis="x", rotation=45)
    # (3) binned deaths — two-weekly NB cells + terminal lower-bound cell
    if "deaths" in pa:
        _band(ax[2], death_mid, deaths_binned_pp, "#8856a7", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[2], death_mid, deaths_binned_pp, "#8856a7", "Posterior PPC")
    ax[2].plot(death_mid, ctx["shared"]["obs_deaths_binned"], "k.-", label="Observed")
    ax[2].plot(
        [death_mid[-1]],
        [ctx["shared"]["obs_deaths_binned"][-1]],
        "rv",
        ms=11,
        label="terminal cell (two-sided, loose disp)",
        zorder=6,
    )
    ax[2].set_title("[SCORED] Deaths per cell — 2-weekly NB + terminal two-sided (IFR)")
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", rotation=45)
    shared = ctx["shared"]
    zones = ctx["zones"]
    obs_zf = ctx["obs_zone_final"]
    # ============================ ROW 1 — SCORED ============================
    # ax[0] cum-confirmed (above), ax[1] 3-day-new confirmed, ax[2] deaths (above).
    # (ROW1d) SCORED — confirmed-in-iso (tracing comparator)
    if "conf_iso" in pa:
        _band(ax[3], iso_dates, confirmed_iso_pp, "#2171b5", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[3], iso_dates, confirmed_iso_pp, "#2171b5", "Posterior PPC")
    ax[3].plot(iso_dates, shared["obs_confirmed_iso"], "k.-", label="Observed")
    ax[3].set_title("[SCORED] Confirmed-in-iso (tracing signal)")
    ax[3].legend(fontsize=8)
    ax[3].tick_params(axis="x", rotation=45)

    # ============================ ROW 2 — SCORED ============================
    # (ROW2a) SCORED — core-vs-periphery 'frac-other' SPLIT.
    top_idx = shared.get("dm_top_idx", np.arange(len(zones)))
    other_mask = shared.get("dm_other_mask", np.zeros(len(zones), bool))
    _sb = shared["split_background"]
    obs_lump = dm_lump(obs_zf, top_idx, other_mask)
    p_obs_other = float(obs_lump[-1] / max(obs_lump.sum(), 1))
    split_pp = np.array(
        [
            dm_lump(zone_final_pp[i] + _sb, top_idx, other_mask)
            for i in range(zone_final_pp.shape[0])
        ]
    )
    split_pp = split_pp / np.maximum(split_pp.sum(axis=1, keepdims=True), 1)
    q_other = np.percentile(split_pp[:, -1], [5, 50, 95])
    ll_split = beta_binomial_loglik(
        float(obs_lump[-1]),
        float(obs_lump.sum()),
        float(q_other[1]),
        shared.get("betabin_s", 300.0),
    )
    ax[4].bar([0], [p_obs_other], width=0.5, color="k", label="Observed 'other' share")
    ax[4].bar(
        [1],
        [q_other[1]],
        width=0.5,
        color="#2171b5",
        alpha=0.85,
        yerr=[[q_other[1] - q_other[0]], [q_other[2] - q_other[1]]],
        capsize=6,
        ecolor="#08519c",
        label="Posterior PPC (median, 5-95%)",
    )
    ax[4].axhline(p_obs_other, color="k", ls="--", lw=0.8, zorder=0)
    ax[4].set_xticks([0, 1])
    ax[4].set_xticklabels(["obs", "model"])
    ax[4].set_xlim(-0.6, 1.6)
    ax[4].set_ylabel("core-vs-periphery 'other' share")
    ax[4].set_title(
        f"[SCORED] frac-other split (Beta-Binom, s={shared.get('betabin_s', 300):.0f})\n"
        f"model {q_other[1]:.3f} vs obs {p_obs_other:.3f} ({q_other[1] / max(p_obs_other, 1e-9):.2f}x), ll {ll_split:.1f}",
        fontsize=10,
    )
    ax[4].legend(fontsize=7)

    # (ROW2b) SCORED — within-core top-15 DM: per-zone spatial zone-final.
    order = np.argsort(-obs_zf)
    n_show = min(20, len(zones))
    sel = order[:n_show]
    xpos = np.arange(n_show)
    qz = np.percentile(zone_final_pp[:, sel], [25, 50, 75], axis=0)
    ax[5].bar(xpos - 0.2, obs_zf[sel], width=0.4, color="k", label="Observed")
    ax[5].bar(
        xpos + 0.2,
        qz[1],
        width=0.4,
        color="#2171b5",
        alpha=0.8,
        label="Posterior PPC (median, 25-75)",
        yerr=[qz[1] - qz[0], qz[2] - qz[1]],
        capsize=3,
        ecolor="#2171b5",
    )
    ax[5].set_xticks(xpos)
    ax[5].set_xticklabels([zones[i] for i in sel], rotation=60, ha="right", fontsize=8)
    _dm_top = int(top_idx.size) or 0
    ax[5].set_title(
        f"[SCORED] within-core DM zone-final\n(top {n_show} of {len(zones)})", fontsize=10
    )
    ax[5].legend(fontsize=7)

    # (ROW2c) SCORED anchor — latent burden (curve is overlay; end-horizon anchor is scored).
    if "lat" in pa:
        _band(ax[6], dates, lat_pp, "#238b45", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[6], dates, lat_pp, "#238b45", "Posterior PPC")
    ql = np.percentile(lat_pp, [5, 50, 95], axis=0)
    epi_md = int(shared.get("epi_estimate", 4500))
    epi_date = shared.get("epi_date", "2026-07-06")
    from scipy.stats import nbinom as _nb_epi

    _epi_disp = float(shared.get("epi_disp", 20.0))
    _p_epi = _epi_disp / (_epi_disp + epi_md)
    epi_lo, epi_hi = _nb_epi.ppf(0.05, _epi_disp, _p_epi), _nb_epi.ppf(0.95, _epi_disp, _p_epi)
    ax[6].errorbar(
        [dates[-1]],
        [epi_md],
        yerr=[[epi_md - epi_lo], [epi_hi - epi_md]],
        fmt="o",
        color="red",
        ms=6,
        capsize=4,
        lw=1.5,
        label=f"epiforecasts {epi_date} 90% ({epi_md / 1000:.1f}k @ horizon)",
        zorder=10,
    )
    ax[6].set_title(
        f"[anchor SCORED | curve overlay]\nLatent burden — end 5-95% {ql[0, -1]:.0f}-{ql[2, -1]:.0f}",
        fontsize=10,
    )
    ax[6].legend(fontsize=8)
    ax[6].tick_params(axis="x", rotation=45)

    # (ROW2d) Bunia backlog (overlay / summary-stat depending on n_scored_bk).
    qq = np.percentile(queue_pp, [5, 50, 95], axis=0)
    ax[7].errorbar(
        bk_dates,
        qq[1],
        yerr=[qq[1] - qq[0], qq[2] - qq[1]],
        fmt="o",
        color="#6a51a3",
        label="Posterior PPC (scored+overlay)",
        capsize=6,
        ms=10,
    )
    ax[7].scatter(
        [d + pd.Timedelta(hours=8) for d in bk_dates_scored],
        shared["obs_backlog"],
        color="k",
        s=120,
        label=f"Documented ({n_scored_bk} scored)",
        zorder=5,
    )
    if len(bk_dates_overlay):
        ax[7].scatter(
            [d + pd.Timedelta(hours=8) for d in bk_dates_overlay],
            shared["overlay_backlog_values"],
            color="k",
            marker="x",
            s=80,
            label="Documented (overlay only)",
            zorder=5,
        )
    _bk_tag = "[summary stat]" if n_scored_bk > 0 else "[overlay]"
    ax[7].set_title(
        f"{_bk_tag} Bunia backlog ({n_scored_bk} scored + {len(bk_dates_overlay)} overlay)"
    )
    ax[7].legend(fontsize=7)
    ax[7].tick_params(axis="x", rotation=45)

    # ============================ ROW 3 — OVERLAYS + text ============================
    # (ROW3a) suspect-in-iso overlay (scaled 1/positivity).
    if "susp_iso" in pa:
        _band(
            ax[8],
            iso_dates,
            suspect_iso_pp / pos,
            "#e6550d",
            f"Posterior PPC (model / {pos:.2f})",
            alpha=0.4,
        )
    else:
        _band(ax[8], iso_dates, suspect_iso_pp / pos, "#e6550d", f"PPC median / {pos:.2f}")
    ax[8].plot(iso_dates, shared["obs_suspect_iso_val"], "k.-", label="Observed suspect")
    ax[8].set_title(f"[overlay] Suspect-in-iso (scaled 1/{pos:.2f})")
    ax[8].legend(fontsize=8)
    ax[8].tick_params(axis="x", rotation=45)

    # (ROW3b) full-window total iso stock overlay.
    if "iso_stock" in pa:
        _band(
            ax[9],
            dates,
            iso_stock_pp,
            "#54278f",
            "Posterior PPC (model TOTAL true-iso)",
            alpha=0.4,
        )
    else:
        _band(ax[9], dates, iso_stock_pp, "#54278f", "Model TOTAL true-iso median")
    if len(ctx["iso_stock_obs"]):
        ax[9].plot(
            ctx["iso_stock_obs"]["date"],
            ctx["iso_stock_obs"]["value"],
            "k.-",
            label="Obs ward census (incl. negatives)",
        )
    ax[9].set_title(
        "[overlay] Full-window iso stock\n(model BDBV+ only; obs ward incl. negatives)",
        fontsize=10,
    )
    ax[9].legend(fontsize=8)
    ax[9].tick_params(axis="x", rotation=45)

    # (ROW3c) recovered flow overlay (tracks data well — kept; admissions dropped).
    if "rec" in pa:
        _band(ax[10], rec_mid, rec_pp, "#41ab5d", "Posterior PPC", alpha=0.4)
    else:
        _band(ax[10], rec_mid, rec_pp, "#41ab5d", "Posterior PPC")
    ax[10].plot(rec_mid, shared["obs_rec_new"], "k.-", label="Observed (cumul guéris)")
    ax[10].set_title("[overlay] 3-day-new recovered (not scored)")
    ax[10].legend(fontsize=8)
    ax[10].tick_params(axis="x", rotation=45)

    # (ROW3d) summary text panel
    ax[11].axis("off")
    ess_str = f"{ess:.1f}" if np.isfinite(ess) else "n/a (regen from combined posterior)"
    ess_over_n_str = f"{ess / N:.5f}" if np.isfinite(ess) else "n/a"
    _epi_md = int(ctx["shared"].get("epi_estimate", 5600))
    _epi_date = str(ctx["shared"].get("epi_date", "2026-07-18"))
    _horizon = (
        pd.Timestamp(ctx["dates"][-1]).date().isoformat()
    )  # actual horizon = last modelled day
    summary_txt = (
        f"{_brake_title()}\n"
        f"{len(CAL)}-param CAL + emcee (horizon {_horizon})\n\n"
        f"N = {N}\nESS = {ess_str} (ESS/N = {ess_over_n_str})\n\n"
        f"Burden end (5/50/95%):\n  {ql[0, -1]:.0f} / {ql[1, -1]:.0f} / {ql[2, -1]:.0f}\n"
        f"  vs epiforecasts anchor {_epi_date} (SCORED):\n"
        f"  NB(mean={_epi_md}, disp={ctx['shared'].get('epi_disp', 20):.0f})\n\n"
        f"Cum confirmed end:\n  obs {ctx['obs_nat_cum'][-1]:.0f},  model med {np.median(nat_pp[:, -1]):.0f}\n\n"
        f"Confirmed-in-iso end (SCORED):\n  obs {ctx['shared']['obs_confirmed_iso'][-1]:.0f},  model med {np.median(confirmed_iso_pp[:, -1]):.0f}\n\n"
        f"Bunia backlog: overlay only (not scored)\n"
        f"  obs 203 (25 May) / 47 (1 Jun),  model med {np.median(queue_pp[:, 0]):.0f} / {np.median(queue_pp[:, 1]):.0f}\n\n"
        f"SCORED spatial: frac-other split + within-core DM cases\n\n"
        f"Panels marked [overlay] are NOT scored:\n"
        f"  latent burden curve (only end-of-horizon anchor scored)\n"
        f"  DM deaths, suspect-in-iso, full-window iso stock,\n"
        f"  admissions, recovered\n"
    )
    ax[11].text(
        0.05,
        0.95,
        summary_txt,
        transform=ax[11].transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )
    ess_title = (
        f"ESS={ess:.0f}, ESS/N={ess / N:.4f}"
        if np.isfinite(ess)
        else "regen from combined posterior"
    )
    # Title derived from the live config, never written as a literal, so it cannot assert a
    # brake design the run did not actually use.
    fig.suptitle(f"{_brake_title()} — ALL observables. N={N}, {ess_title}", fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(FIG / "00_all_in_one.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_scored_only(
    ctx,
    conf_new_pp,
    deaths_binned_pp,
    zone_final_pp,
    zone_deaths_final_pp,
    queue_pp,
    confirmed_iso_pp,
    lat_pp,
    adm_pp,
    rec_pp,
    ess,
    N,
    prior_arrays=None,
) -> None:
    """Focused figure showing ONLY the streams that enter the composite likelihood,
    plus the two DM streams as clearly-labeled overlays.

    adm/rec/suspect_iso are NOT scored. Deaths enter as a two-weekly NB + terminal
    rail, and per-zone DM-DEATHS is not scored. The spatial likelihood is a
    core-vs-periphery Beta-Binomial 'frac-other' SPLIT cell plus the within-core
    top-15 Dirichlet-multinomial (dm_core). Scored
    streams: (1) 3-day-new confirmed, (2) binned deaths (2-weekly NB + terminal LB),
    (3) confirmed-in-iso, (4) core-vs-periphery 'frac-other' split, (6) epi-joint
    latent-burden anchor at horizon. OVERLAYS (not scored): (5) within-core top-15
    DM cases, (9) DM deaths."""
    conf_edges = ctx["conf_edges"]
    conf_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(conf_edges[i + 1]))
        for i in range(len(conf_edges) - 1)
    ]
    death_edges = ctx["shared"]["death_edges"]
    death_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(death_edges[i + 1]))
        for i in range(len(death_edges) - 1)
    ]
    isoval = ctx["isoval"]
    iso_dates = list(isoval["date"])
    n_scored_bk = ctx["shared"]["n_scored_backlog"]
    shared = ctx["shared"]

    fig, ax = plt.subplots(3, 3, figsize=(21, 15))
    ax = ax.ravel()

    # (1) SCORED — 3-day-new confirmed
    _band(ax[0], conf_mid, conf_new_pp, "#d95f0e", "Posterior PPC")
    ax[0].plot(conf_mid, shared["obs_conf_new"], "k.-", label="Observed")
    ax[0].set_title(f"SCORED: 3-day-new confirmed ({len(conf_mid)} cells)")
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", rotation=45)

    # (2) SCORED — binned deaths (two-weekly NB cells + terminal lower-bound cell)
    _band(ax[1], death_mid, deaths_binned_pp, "#8856a7", "Posterior PPC")
    ax[1].plot(death_mid, shared["obs_deaths_binned"], "k.-", label="Observed")
    ax[1].plot(
        [death_mid[-1]],
        [shared["obs_deaths_binned"][-1]],
        "rv",
        ms=11,
        label="terminal cell (two-sided, loose disp)",
        zorder=6,
    )
    _n_nb = int(shared["n_death_nb_cells"])
    ax[1].set_title(f"SCORED: deaths ({_n_nb} two-weekly NB + 1 terminal two-sided)")
    ax[1].legend(fontsize=8)
    ax[1].tick_params(axis="x", rotation=45)

    # (3) SCORED — confirmed-in-iso census
    _band(ax[2], iso_dates, confirmed_iso_pp, "#2171b5", "Posterior PPC")
    ax[2].plot(iso_dates, shared["obs_confirmed_iso"], "k.-", label="Observed")
    ax[2].set_title(f"SCORED: confirmed-in-iso ({len(iso_dates)} cells)")
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", rotation=45)

    # (4) SCORED — spatial core-vs-periphery 'frac-other' SPLIT (Beta-Binomial).
    # The ONLY spatial term in the composite likelihood is a single
    # Beta-Binomial cell on the observed 'other' count vs the model's 'other' share
    # (forecast-critical leakage-out-of-core). The within-core top-15 DM is an
    # OVERLAY (panel 5), not scored. Show obs vs PPC 'other' share with the 5-95 band.
    zones = ctx["zones"]
    obs_zf = ctx["obs_zone_final"]
    top_idx = shared.get("dm_top_idx", np.arange(len(zones)))
    other_mask = shared.get("dm_other_mask", np.zeros(len(zones), bool))
    _dm_n = int(top_idx.size)
    labels = [zones[i] for i in top_idx] + ["other"]
    xpos = np.arange(len(labels))
    obs_share = dm_lump(obs_zf, top_idx, other_mask)
    obs_share = obs_share / max(obs_share.sum(), 1)
    # The SCORED-split share and ll_split MUST use split_background so this panel matches
    # `log_likelihood`; plotting the raw no-background share beside a background-inflated
    # title log-lik makes the bar and the title disagree. `lumped_pp` (no bg) is only for
    # the within-core overlay panels below, which are unscored.
    _sb = shared["split_background"]
    lumped_pp = np.array(
        [dm_lump(zone_final_pp[i], top_idx, other_mask) for i in range(zone_final_pp.shape[0])]
    )
    lumped_pp = lumped_pp / np.maximum(lumped_pp.sum(axis=1, keepdims=True), 1)
    qm = np.percentile(lumped_pp, [5, 50, 95], axis=0)
    split_pp = np.array(
        [
            dm_lump(zone_final_pp[i] + _sb, top_idx, other_mask)
            for i in range(zone_final_pp.shape[0])
        ]
    )
    split_pp = split_pp / np.maximum(split_pp.sum(axis=1, keepdims=True), 1)
    p_obs_other = float(obs_share[-1])
    q_other = np.percentile(split_pp[:, -1], [5, 50, 95])
    # Beta-Binomial log-lik at the PPC-median 'other' share (now genuinely matches log_likelihood)
    obs_lump = dm_lump(obs_zf, top_idx, other_mask)
    ll_split = beta_binomial_loglik(
        float(obs_lump[-1]),
        float(obs_lump.sum()),
        float(q_other[1]),
        shared.get("betabin_s", 300.0),
    )
    ax[3].bar([0], [p_obs_other], width=0.5, color="k", label="Observed 'other' share")
    ax[3].bar(
        [1],
        [q_other[1]],
        width=0.5,
        color="#2171b5",
        alpha=0.85,
        yerr=[[q_other[1] - q_other[0]], [q_other[2] - q_other[1]]],
        capsize=6,
        ecolor="#08519c",
        label="Posterior PPC (median, 5-95%)",
    )
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["obs", "model"])
    ax[3].set_xlim(-0.6, 1.6)
    ax[3].set_ylabel("core-vs-periphery 'other' share")
    ax[3].set_title(
        f"SCORED: core/periphery split (1 Beta-Binomial cell, s={shared.get('betabin_s', 300):.0f}, log-lik {ll_split:.1f})"
    )
    ax[3].legend(fontsize=8)

    # (5) OVERLAY: within-core top-15 DM CASES share, its own
    # panel so it reads distinctly from the scored frac-other split in panel 4. This
    # is the full top-15 Dirichlet-multinomial. DM-deaths gets its own overlay panel (8).
    ax[4].bar(xpos - 0.2, obs_share, width=0.4, color="k", label="Observed share")
    ax[4].bar(
        xpos + 0.2,
        qm[1],
        width=0.4,
        color="#41ab5d",
        alpha=0.85,
        yerr=[qm[1] - qm[0], qm[2] - qm[1]],
        capsize=3,
        ecolor="#238b45",
        label="Posterior PPC (median, 5-95%)",
    )
    ax[4].set_xticks(xpos)
    ax[4].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax[4].set_ylabel("share")
    ax[4].set_title(
        f"SCORED: within-core top-{_dm_n} DM cases (\u03b1={shared.get('dm_alpha_cases', 30):.0f})"
    )
    ax[4].legend(fontsize=8)

    # (6) SCORED — epi_joint anchor (single cell at horizon)
    epi_md = int(shared.get("epi_estimate", 4800))
    epi_disp = float(shared.get("epi_disp", 20.0))
    epi_date_str = str(shared.get("epi_date", "2026-07-08"))
    # NB 5-95 for the anchor given its own disp
    from scipy.stats import nbinom as _nb

    # NB param: mean=epi_md, disp=epi_disp -> p = disp/(disp+mean)
    p_nb = epi_disp / (epi_disp + epi_md)
    anchor_lo, anchor_hi = _nb.ppf(0.05, epi_disp, p_nb), _nb.ppf(0.95, epi_disp, p_nb)
    model_end = lat_pp[:, -1] if lat_pp.size else np.array([np.nan])
    q_end = np.percentile(model_end, [5, 50, 95])
    ax[5].errorbar(
        [0],
        [q_end[1]],
        yerr=[[q_end[1] - q_end[0]], [q_end[2] - q_end[1]]],
        fmt="o",
        color="#2171b5",
        ms=10,
        capsize=6,
        label="Model latent @ horizon\n(median [5-95%])",
    )
    ax[5].errorbar(
        [1],
        [epi_md],
        yerr=[[epi_md - anchor_lo], [anchor_hi - epi_md]],
        fmt="o",
        color="red",
        ms=10,
        capsize=6,
        label=f"Anchor {epi_date_str}\nNB(mean={epi_md}, disp={epi_disp:.0f})",
    )
    ax[5].set_xticks([0, 1])
    ax[5].set_xticklabels(["Model", "Anchor"])
    ax[5].set_xlim(-0.5, 1.5)
    ax[5].set_ylabel("Cumulative infections")
    ax[5].set_title("SCORED: epi_joint anchor (1 cell, latent burden at horizon)")
    ax[5].legend(fontsize=8)

    # (7) SCORED — backlog (only if any cells scored; else omit)
    if n_scored_bk > 0:
        scored_bk_dates = [pd.Timestamp(d) for d in ["2026-05-25", "2026-06-01"][:n_scored_bk]]
        qq = np.percentile(queue_pp, [5, 50, 95], axis=0)
        ax[6].errorbar(
            scored_bk_dates,
            qq[1, :n_scored_bk],
            yerr=[
                qq[1, :n_scored_bk] - qq[0, :n_scored_bk],
                qq[2, :n_scored_bk] - qq[1, :n_scored_bk],
            ],
            fmt="o",
            color="#6a51a3",
            capsize=6,
            ms=11,
            label="Posterior PPC",
        )
        ax[6].scatter(
            scored_bk_dates,
            shared["obs_backlog"][:n_scored_bk],
            color="k",
            s=120,
            label="Documented",
            zorder=5,
        )
        ax[6].set_title(f"SCORED: backlog ({n_scored_bk} cells)")
        ax[6].legend(fontsize=8)
        ax[6].tick_params(axis="x", rotation=45)
    else:
        ax[6].axis("off")
        ax[6].text(
            0.5,
            0.5,
            "backlog: 0 scored cells\n(overlay-only in config)",
            ha="center",
            va="center",
            transform=ax[6].transAxes,
            fontsize=11,
            color="grey",
        )

    # (9) OVERLAY: DM DEATHS share, its own panel, dropped from
    # the likelihood as spatially unreliable (deaths reshuffle between zones). Kept for
    # context only — never enters the composite likelihood.
    obs_zd = shared["obs_zone_deaths_final"]
    obs_d_share = dm_lump(obs_zd, top_idx, other_mask)
    obs_d_share = obs_d_share / max(obs_d_share.sum(), 1)
    lumped_d_pp = np.array(
        [
            dm_lump(zone_deaths_final_pp[i], top_idx, other_mask)
            for i in range(zone_deaths_final_pp.shape[0])
        ]
    )
    lumped_d_pp = lumped_d_pp / np.maximum(lumped_d_pp.sum(axis=1, keepdims=True), 1)
    qd = np.percentile(lumped_d_pp, [5, 50, 95], axis=0)
    ax[8].bar(xpos - 0.2, obs_d_share, width=0.4, color="k", label="Observed share")
    ax[8].bar(
        xpos + 0.2,
        qd[1],
        width=0.4,
        color="#feb24c",
        alpha=0.85,
        yerr=[qd[1] - qd[0], qd[2] - qd[1]],
        capsize=3,
        ecolor="#d95f0e",
        label="Posterior PPC (median, 5-95%)",
    )
    ax[8].set_xticks(xpos)
    ax[8].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax[8].set_ylabel("share")
    ax[8].set_title("OVERLAY — not scored: DM deaths")
    ax[8].legend(fontsize=8)

    # (8) summary text
    ax[7].axis("off")
    # The scored spatial term is ONE Beta-Binomial 'frac-other' cell.
    # The 16-cell within-core DM is an OVERLAY (dm_core), not part of the total.
    split_cells = 1
    n_death_cells = len(death_mid)  # n_death_nb_cells NB + 1 terminal lower-bound
    n_nb = int(shared["n_death_nb_cells"])
    total_cells = len(conf_mid) + n_death_cells + n_scored_bk + len(iso_dates) + split_cells + 1
    ess_str = f"{ess:.1f}" if np.isfinite(ess) else "n/a (regen)"
    ess_over_n_str = f"{ess / N:.5f}" if np.isfinite(ess) else "n/a"
    summary_txt = (
        f"SCORED streams ONLY (composite log-likelihood inputs)\n\n"
        f"NOT scored — panels 4/5 & 00_all_in_one show overlays:\n"
        f"  DM deaths\n"
        f"  latent burden curve  (only endpoint anchor scored)\n"
        f"  suspect-in-iso census\n"
        f"  full-window total iso stock\n"
        f"  3-day admissions\n"
        f"  3-day recovered\n\n"
        f"Cells in composite likelihood:\n"
        f"  3-day-new confirmed:   {len(conf_mid):3d}    NB(k={shared['disp']:.2f})\n"
        f"  deaths (2wk NB+term LB):{n_death_cells:3d}    {n_nb} NB + 1 lower-bound\n"
        f"  confirmed-in-iso:      {len(iso_dates):3d}    NB(k={shared['disp']:.2f})\n"
        f"  backlog:               {n_scored_bk:3d}\n"
        f"  core/periphery split:  {split_cells:3d}    Beta-Binomial(s={shared.get('betabin_s', 300):.0f})\n"
        f"  epi_joint anchor:        1    NB(k={shared.get('epi_disp', 20):.0f})\n"
        f"  ─────────\n"
        f"  TOTAL cells:           {total_cells:3d}\n\n"
        f"N = {N}\nESS = {ess_str}  ESS/N = {ess_over_n_str}\n"
    )
    ax[7].text(
        0.02,
        0.98,
        summary_txt,
        transform=ax[7].transAxes,
        fontsize=9.5,
        verticalalignment="top",
        family="monospace",
    )
    fig.suptitle(
        f"{_brake_title()} — Scored streams (composite log-likelihood inputs) + DM overlays",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(FIG / "05_scored_only.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_likelihood_diagnosis(
    ctx,
    conf_new_pp,
    deaths_binned_pp,
    confirmed_iso_pp,
    zone_final_pp,
    zone_deaths_final_pp,
    lat_pp,
    ess,
    N,
) -> None:
    """Per-component likelihood diagnosis (SCORED streams only; DM as overlay).

    Answers 'which SCORED components are fitting vs not, piece by piece'. The
    scored components match `log_likelihood`: conf_new, deaths (NB+LB),
    confirmed_iso, the core-vs-periphery Beta-Binomial 'frac-other' SPLIT, and the
    epi_joint anchor. The within-core top-15 Dirichlet-multinomial (dm_core, moved
    to overlay) and DM-deaths (not scored) are computed for
    context and drawn in a clearly-labeled overlay panel, but are NEVER summed into
    the ranks or TOTAL. Computed at the posterior median of each PPC output (an
    approximation of the likelihood the calibrator saw): (a) aggregate
    log-likelihood magnitude per scored component, so we see which streams
    dominate; (b) per-cell normalised residuals (obs - PPC_med) / half-IQR, so a
    value near 0 = well-fit and abs > 1 = the cell sits outside the model's 25-75
    posterior — a fit signal.
    """
    from scipy.stats import nbinom as _nb

    shared = ctx["shared"]
    disp = float(shared["disp"])

    conf_edges = ctx["conf_edges"]
    conf_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(conf_edges[i + 1]))
        for i in range(len(conf_edges) - 1)
    ]
    death_edges = shared["death_edges"]
    death_mid = [
        pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(death_edges[i + 1]))
        for i in range(len(death_edges) - 1)
    ]
    isoval = ctx["isoval"]
    iso_dates = list(isoval["date"])

    def _nb_ll_per_cell(obs, model_med):
        mu = np.maximum(model_med, 1e-6)
        p = disp / (disp + mu)
        return _nb.logpmf(obs, disp, p)

    def _iqr_residual(obs, pp_arr):
        q = np.percentile(pp_arr, [25, 50, 75], axis=0)
        half_iqr = np.maximum((q[2] - q[0]) / 2.0, 1e-6)
        return (obs - q[1]) / half_iqr, q[1]

    # --- aggregate log-lik per component ---
    conf_med = np.median(conf_new_pp, axis=0)
    deaths_med = np.median(deaths_binned_pp, axis=0)
    iso_med = np.median(confirmed_iso_pp, axis=0)
    zf_med = np.median(zone_final_pp, axis=0)
    zd_med = np.median(zone_deaths_final_pp, axis=0)

    obs_conf_new = shared["obs_conf_new"]
    obs_deaths_binned = shared["obs_deaths_binned"]
    obs_confirmed_iso = shared["obs_confirmed_iso"]

    ll_conf = _nb_ll_per_cell(obs_conf_new, conf_med)
    # per-cell NB for the leading cells; terminal cell is a soft lower bound
    # (no penalty where the model meets/exceeds the under-counted target).
    n_nb = int(shared["n_death_nb_cells"])
    ll_deaths = _nb_ll_per_cell(obs_deaths_binned, deaths_med)
    if n_nb < len(obs_deaths_binned) and max(deaths_med[-1], 1e-6) >= obs_deaths_binned[-1]:
        ll_deaths[-1] = 0.0
    ll_iso = _nb_ll_per_cell(obs_confirmed_iso, iso_med)

    top_idx = shared.get("dm_top_idx")
    other_mask = shared.get("dm_other_mask")
    obs_zc_lump = dm_lump(ctx["obs_zone_final"], top_idx, other_mask)
    obs_zd_lump = dm_lump(shared["obs_zone_deaths_final"], top_idx, other_mask)
    from ..model.likelihood import beta_binomial_loglik, dirichlet_multinomial_loglik

    # SCORED spatial term: a single
    # Beta-Binomial cell on the CORE-vs-PERIPHERY split. `k` = observed 'other'
    # count, `n` = observed scored total, `p_model` = the model's 'other' share;
    # concentration `betabin_s`. This is the ONLY spatial term in the composite
    # likelihood — the forecast-critical "how fast does it leak out of the core"
    # quantity. Computed at the PPC-median 'other' share to mirror the other bars.
    # SPLIT share/log-lik use split_background (matches log_likelihood);
    # the dm_core OVERLAY keeps the full `background`. `mod_split_c` drives the SCORED
    # split panel bar so plotted == scored; without it the bar shows the no-background
    # share (~0.024) while the title log-lik uses the background-inflated one (~0.077).
    mod_split_c = dm_lump(zf_med + shared["split_background"], top_idx, other_mask)
    mod_lump_c = dm_lump(
        zf_med + shared["background"], top_idx, other_mask
    )  # overlay dm_core only
    _mtot = float(mod_split_c.sum())
    p_other_model = (float(mod_split_c[-1]) / _mtot) if _mtot > 0 else 0.0
    ll_split = beta_binomial_loglik(
        float(obs_zc_lump[-1]),
        float(obs_zc_lump.sum()),
        p_other_model,
        shared.get("betabin_s", 300.0),
    )

    # dm_core (within-core top-15 Dirichlet-multinomial) IS SCORED
    # (see log_likelihood: `base_ll += split_ll + core_ll`), so it belongs in
    # the SCORED bars/total below — NOT the overlay. DM-deaths remains dropped.
    ll_dm_core = dirichlet_multinomial_loglik(
        obs_zc_lump[:-1], mod_lump_c[:-1], shared.get("dm_alpha_cases", 30.0)
    )
    ll_dm_deaths = dirichlet_multinomial_loglik(
        obs_zd_lump,
        dm_lump(zd_med + shared["background"], top_idx, other_mask),
        shared.get("dm_alpha_deaths", 10.0),
    )

    epi_md = int(shared.get("epi_estimate", 4800))
    epi_disp = float(shared.get("epi_disp", 20.0))
    lat_end_med = float(np.median(lat_pp[:, -1])) if lat_pp.size else 1.0
    p_epi = epi_disp / (epi_disp + max(lat_end_med, 1))
    ll_epi = float(_nb.logpmf(epi_md, epi_disp, p_epi))

    # Bar chart data — SCORED components only, matching `log_likelihood`:
    # conf_new, deaths (NB+LB), confirmed_iso, spatial SPLIT (Beta-Binomial frac-other),
    # within-core DM (dm_core), epi_joint. DM-deaths remains an
    # overlay and is plotted separately below, never summed here.
    comp_labels = [
        "conf_new\n(3-day)",
        "deaths\n(2wk NB+LB)",
        "confirmed_iso\n(weekly)",
        "split\n(frac-other BB)",
        "dm_core\n(within-core DM)",
        "epi_joint\n(1 anchor)",
    ]
    comp_ll = np.array(
        [ll_conf.sum(), ll_deaths.sum(), ll_iso.sum(), ll_split, ll_dm_core, ll_epi]
    )
    comp_cells = np.array(
        [len(obs_conf_new), len(obs_deaths_binned), len(obs_confirmed_iso), 1, 1, 1]
    )
    comp_per_cell = comp_ll / np.maximum(comp_cells, 1)
    colors = ["#d95f0e", "#8856a7", "#2171b5", "#41ab5d", "#e31a1c"]

    fig, ax = plt.subplots(3, 3, figsize=(21, 15))

    # (0) Aggregate log-lik magnitude per component
    xpos = np.arange(len(comp_labels))
    ax[0, 0].bar(xpos, comp_ll, color=colors)
    ax[0, 0].set_xticks(xpos)
    ax[0, 0].set_xticklabels(comp_labels, fontsize=8)
    ax[0, 0].axhline(0, color="k", lw=0.5)
    for i, (v, n) in enumerate(zip(comp_ll, comp_cells)):
        ax[0, 0].text(
            i, v, f"{v:.0f}\n({n}c)", ha="center", va="bottom" if v < 0 else "top", fontsize=8
        )
    ax[0, 0].set_title("Aggregate log-lik per SCORED component (at PPC median)")
    ax[0, 0].set_ylabel("sum log-lik")

    # (1) Per-cell log-lik magnitude (more negative = worse fit)
    ax[0, 1].bar(xpos, comp_per_cell, color=colors)
    ax[0, 1].set_xticks(xpos)
    ax[0, 1].set_xticklabels(comp_labels, fontsize=8)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    for i, v in enumerate(comp_per_cell):
        ax[0, 1].text(i, v, f"{v:.1f}", ha="center", va="bottom" if v < 0 else "top", fontsize=9)
    ax[0, 1].set_title("Log-lik PER CELL, SCORED (fair per-observation comparison)")
    ax[0, 1].set_ylabel("log-lik / cell")

    # (2) Text summary
    ax[0, 2].axis("off")
    tot = comp_ll.sum()
    txt = (
        "Interpretation:\n\n"
        "'PER CELL' bars show which SCORED streams the calibrator\n"
        "finds hardest — a more negative value means each\n"
        "observation of that stream costs the posterior more to\n"
        "fit. TWO spatial cells are SCORED: the\n"
        "Beta-Binomial 'frac-other' SPLIT (core vs periphery)\n"
        "and the within-core top-15 DM (dm_core). Only\n"
        "DM-deaths is an overlay, NOT scored.\n\n"
        "The residual plots below show cell-by-cell whether\n"
        "obs sits inside the posterior 25-75 IQR (|z| < 1) or\n"
        "outside (|z| > 1 = a fit signal).\n\n"
        "SCORED component ranks (by total log-lik):\n"
    )
    order = np.argsort(comp_ll)[::-1]  # least-negative first
    for i in order:
        txt += f"  {comp_labels[i].replace(chr(10), ' '):24s}  {comp_ll[i]:+8.1f}\n"
    txt += f"\n  SCORED TOTAL:            {tot:+8.1f}\n\n"
    txt += f"OVERLAY (not scored — for context only):\n  DM-deaths   {ll_dm_deaths:+8.1f}\n"
    ax[0, 2].text(
        0.02,
        0.98,
        txt,
        transform=ax[0, 2].transAxes,
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
    )

    # (3) Per-cell residuals — 3-day new confirmed
    z, _ = _iqr_residual(obs_conf_new, conf_new_pp)
    ax[1, 0].axhspan(-1, 1, color="grey", alpha=0.2)
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].plot(conf_mid, z, "o-", color="#d95f0e")
    ax[1, 0].set_ylabel("(obs - PPC med) / half-IQR")
    ax[1, 0].set_title("conf_new residuals   |z|>1 outside PPC IQR")
    ax[1, 0].tick_params(axis="x", rotation=45)
    ax[1, 0].set_ylim(-4, 4)

    # (4) Per-cell residuals — binned deaths (last point = terminal lower-bound cell)
    z, _ = _iqr_residual(obs_deaths_binned, deaths_binned_pp)
    ax[1, 1].axhspan(-1, 1, color="grey", alpha=0.2)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].plot(death_mid, z, "o-", color="#8856a7")
    ax[1, 1].plot([death_mid[-1]], [z[-1]], "rv", ms=10, zorder=6, label="terminal two-sided cell")
    ax[1, 1].set_title("deaths residuals (2wk NB + terminal two-sided)")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].tick_params(axis="x", rotation=45)
    ax[1, 1].set_ylim(-4, 4)

    # (5) Per-cell residuals — confirmed_iso
    z, _ = _iqr_residual(obs_confirmed_iso, confirmed_iso_pp)
    ax[1, 2].axhspan(-1, 1, color="grey", alpha=0.2)
    ax[1, 2].axhline(0, color="k", lw=0.5)
    ax[1, 2].plot(iso_dates, z, "o-", color="#2171b5")
    ax[1, 2].set_title("confirmed_iso residuals")
    ax[1, 2].tick_params(axis="x", rotation=45)
    ax[1, 2].set_ylim(-4, 4)

    # (6) SCORED spatial term — core-vs-periphery Beta-Binomial 'frac-other' SPLIT.
    # This is the ONLY spatial cell in the composite likelihood. Show
    # the observed vs PPC 'other' share directly (the quantity the BB scores), with
    # the PPC 5-95 band so the fit is honest about spread, not just the median.
    labels_lump = [ctx["zones"][i] for i in top_idx] + ["other"]
    xp = np.arange(len(labels_lump))
    # The SCORED bar must use the SAME split_background as the scored 'other' share and
    # the ll_split title; otherwise a raw no-background share (~0.024) is plotted beside
    # a background-inflated title log-lik (~0.077) and the two disagree.
    _sb = shared["split_background"]
    lump_pp_c = np.array(
        [
            dm_lump(zone_final_pp[i] + _sb, top_idx, other_mask)
            for i in range(zone_final_pp.shape[0])
        ]
    )
    lump_pp_c = lump_pp_c / np.maximum(lump_pp_c.sum(axis=1, keepdims=True), 1)
    p_obs_other = float(obs_zc_lump[-1]) / max(float(obs_zc_lump.sum()), 1)
    other_share_pp = lump_pp_c[:, -1]
    q_other = np.percentile(other_share_pp, [5, 50, 95])
    ax[2, 0].bar([0], [p_obs_other], width=0.5, color="k", label="Observed 'other' share")
    ax[2, 0].bar(
        [1],
        [q_other[1]],
        width=0.5,
        color="#41ab5d",
        alpha=0.85,
        yerr=[[q_other[1] - q_other[0]], [q_other[2] - q_other[1]]],
        capsize=6,
        ecolor="#238b45",
        label="PPC 'other' share (median, 5-95%)",
    )
    ax[2, 0].set_xticks([0, 1])
    ax[2, 0].set_xticklabels(["obs", "model"])
    ax[2, 0].set_xlim(-0.6, 1.6)
    ax[2, 0].set_ylabel("core-vs-periphery 'other' share")
    ax[2, 0].set_title(
        f"SCORED: core/periphery split  (Beta-Binomial log-lik {ll_split:.1f}, s={shared.get('betabin_s', 300):.0f})"
    )
    ax[2, 0].legend(fontsize=8)

    # (7) OVERLAY: within-core top-15 DM CASES share, its own
    # panel so it reads distinctly from the scored frac-other split in (2,0). This is
    # the full top-15 Dirichlet-multinomial. DM-deaths (also unscored) is reported
    # numerically in the (0,2) overlay text block rather than crowding this panel.
    labels_lump = [ctx["zones"][i] for i in top_idx] + ["other"]
    xp = np.arange(len(labels_lump))
    N_cases = obs_zc_lump.sum()
    p_obs_c = obs_zc_lump / max(N_cases, 1)
    p_med_c = np.median(lump_pp_c, axis=0)
    ax[2, 1].bar(xp - 0.2, p_obs_c, width=0.4, color="k", label="Observed")
    ax[2, 1].bar(xp + 0.2, p_med_c, width=0.4, color="#41ab5d", alpha=0.85, label="PPC median")
    ax[2, 1].set_xticks(xp)
    ax[2, 1].set_xticklabels(labels_lump, rotation=60, ha="right", fontsize=8)
    ax[2, 1].set_ylabel("share")
    ax[2, 1].set_title(
        f"SCORED: within-core top-{top_idx.size} DM cases  (dm_core log-lik {ll_dm_core:.0f})"
    )
    ax[2, 1].legend(fontsize=8)

    # (8) Epi anchor
    q_end = np.percentile(lat_pp[:, -1], [5, 50, 95]) if lat_pp.size else np.array([np.nan] * 3)
    p_nb_e = epi_disp / (epi_disp + epi_md)
    a_lo, a_hi = _nb.ppf(0.05, epi_disp, p_nb_e), _nb.ppf(0.95, epi_disp, p_nb_e)
    ax[2, 2].errorbar(
        [0],
        [q_end[1]],
        yerr=[[q_end[1] - q_end[0]], [q_end[2] - q_end[1]]],
        fmt="o",
        color="#2171b5",
        ms=10,
        capsize=6,
        label=f"Model @ horizon\n({q_end[1]:.0f} [5-95%])",
    )
    ax[2, 2].errorbar(
        [1],
        [epi_md],
        yerr=[[epi_md - a_lo], [a_hi - epi_md]],
        fmt="o",
        color="red",
        ms=10,
        capsize=6,
        label=f"Anchor {shared.get('epi_date', '2026-07-08')}\n({epi_md} NB disp {epi_disp:.0f})",
    )
    ax[2, 2].set_xticks([0, 1])
    ax[2, 2].set_xticklabels(["Model", "Anchor"])
    ax[2, 2].set_xlim(-0.5, 1.5)
    ax[2, 2].set_ylabel("Cumulative infections")
    ax[2, 2].set_title(f"epi_joint anchor   (log-lik {ll_epi:.2f})")
    ax[2, 2].legend(fontsize=8)

    fig.suptitle(
        f"{_brake_title()} — Likelihood diagnosis (SCORED components; DM "
        f"streams shown as overlays). N={N}, ESS={ess:.0f}",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(FIG / "06_likelihood_diagnosis.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_spatial(ctx, zone_final_pp) -> None:
    """Detailed spatial PPC — bar chart of observed zone-final vs PPC median+5-95% for all
    calibrated zones, sorted by observed count. Plus residual scatter."""
    zones = ctx["zones"]
    obs_zf = ctx["obs_zone_final"]
    Z = len(zones)
    order = np.argsort(-obs_zf)
    qz = np.percentile(zone_final_pp[:, order], [25, 50, 75], axis=0)  # 25-75 IQR (was 5-95)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    xpos = np.arange(Z)
    ax[0].bar(xpos - 0.2, obs_zf[order], width=0.4, color="k", label="Observed")
    ax[0].bar(
        xpos + 0.2,
        qz[1],
        width=0.4,
        color="#2171b5",
        alpha=0.8,
        label="PPC median (25-75)",
        yerr=[qz[1] - qz[0], qz[2] - qz[1]],
        capsize=2,
        ecolor="#2171b5",
    )
    ax[0].set_xticks(xpos)
    ax[0].set_xticklabels([zones[i] for i in order], rotation=80, ha="right", fontsize=7)
    ax[0].set_title(
        f"Zone-final confirmed (all {Z} calibrated zones, sorted by observed; linear scale)"
    )  # log→linear
    ax[0].legend()
    ax[0].grid(alpha=0.3, axis="y")
    # residual scatter: model_median - obs vs obs
    resid = qz[1] - obs_zf[order]
    ax[1].scatter(obs_zf[order], resid, s=40, alpha=0.7, color="#2171b5")
    for i, idx in enumerate(order[:8]):
        ax[1].annotate(zones[idx], (obs_zf[idx], resid[i]), fontsize=8, alpha=0.7)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xlabel("Observed zone-final")
    ax[1].set_ylabel("PPC median − observed")
    ax[1].set_title("Spatial residuals (top-8 zones labelled)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "04_spatial.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_ppc(ctx, nat_pp, lat_pp, conf_new_pp, post, ess, N) -> None:
    dates = ctx["dates"]
    obs = ctx["obs_nat_cum"]
    edges = ctx["conf_edges"]
    edge_dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(d)) for d in edges]
    mid_dates = [
        edge_dates[i + 1] for i in range(len(edge_dates) - 1)
    ]  # right edge of each window
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
    q = np.percentile(nat_pp, [5, 50, 95], axis=0)
    ax[0].fill_between(dates, q[0], q[2], alpha=0.3, color="#2171b5", label="PPC 5-95%")
    ax[0].plot(dates, q[1], color="#2171b5", lw=2, label="PPC median")
    ax[0].plot(dates, obs, "k.-", label="Observed confirmed")
    ax[0].set_title(f"Cumulative confirmed (daily); ESS/N={ess / N:.3f}, N={N}")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=45)
    qcn = np.percentile(conf_new_pp, [5, 50, 95], axis=0)
    ax[1].fill_between(mid_dates, qcn[0], qcn[2], alpha=0.3, color="#d95f0e", label="PPC 5-95%")
    ax[1].plot(mid_dates, qcn[1], color="#d95f0e", lw=2, label="PPC median")
    ax[1].plot(mid_dates, ctx["shared"]["obs_conf_new"], "k.-", label="Observed 3-day-new")
    ax[1].set_title("3-day-new confirmed (the temporal identifier)")
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=45)
    ql = np.percentile(lat_pp, [5, 50, 95], axis=0)
    ax[2].fill_between(dates, ql[0], ql[2], alpha=0.3, color="#238b45", label="PPC 5-95%")
    ax[2].plot(dates, ql[1], color="#238b45", lw=2, label="PPC median")
    ax[2].set_title(f"Latent burden (all infections)\n5-95% end: {ql[0, -1]:.0f}-{ql[2, -1]:.0f}")
    ax[2].legend()
    ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / "01_ppc_national.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_capacity_iso(
    ctx, queue_pp, confirmed_iso_pp, suspect_iso_pp, iso_stock_pp, post, N
) -> None:
    """Capacity backlog, CONFIRMED-in-iso (summary stat, no scaling), and overlays."""
    dates = ctx["dates"]
    isoval = ctx["isoval"]
    iso_dates = list(isoval["date"])
    n_scored_bk = ctx["shared"]["n_scored_backlog"]
    bk_dates_scored = [pd.Timestamp(d) for d in ["2026-05-25", "2026-06-01"][:n_scored_bk]]
    bk_dates_overlay = [pd.Timestamp(d) for d in ctx["shared"]["overlay_backlog_dates"]]
    bk_dates = bk_dates_scored + bk_dates_overlay
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax = ax.ravel()
    # 1) capacity backlog
    qq = np.percentile(queue_pp, [5, 50, 95], axis=0)
    ax[0].errorbar(
        bk_dates,
        qq[1],
        yerr=[qq[1] - qq[0], qq[2] - qq[1]],
        fmt="o",
        color="#6a51a3",
        label="PPC 5-50-95% (model queue)",
        capsize=4,
    )
    ax[0].scatter(
        bk_dates_scored,
        ctx["shared"]["obs_backlog"],
        color="k",
        s=80,
        label=f"Documented ({n_scored_bk} scored)",
        zorder=5,
    )
    if len(bk_dates_overlay):
        ax[0].scatter(
            bk_dates_overlay,
            ctx["shared"]["overlay_backlog_values"],
            color="k",
            marker="x",
            s=70,
            label="Documented (overlay)",
            zorder=5,
        )
    ax[0].set_title(f"Capacity backlog ({n_scored_bk} scored + {len(bk_dates_overlay)} overlay)")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=45)
    # 2) CONFIRMED-in-iso (summary stat, no scaling) — the clean tracing comparator
    qc = np.percentile(confirmed_iso_pp, [5, 50, 95], axis=0)
    ax[1].fill_between(iso_dates, qc[0], qc[2], alpha=0.3, color="#2171b5", label="PPC 5-95%")
    ax[1].plot(iso_dates, qc[1], color="#2171b5", lw=2, label="PPC median")
    ax[1].plot(
        iso_dates, ctx["shared"]["obs_confirmed_iso"], "k.-", label="Observed (SitRep N°18-N°39)"
    )
    ax[1].set_title(
        "Confirmed-in-isolation [SUMMARY STAT, no scaling]\nTracing-discriminating signal"
    )
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=45)
    # 3) suspect-in-iso overlay (with positivity scaling)
    qs = np.percentile(suspect_iso_pp, [5, 50, 95], axis=0)
    pos = ctx["shared"]["suspect_positivity"]
    ax[2].fill_between(
        iso_dates,
        qs[0] / pos,
        qs[2] / pos,
        alpha=0.3,
        color="#e6550d",
        label=f"PPC 5-95% (model unconfirmed-iso / {pos:.2f})",
    )
    ax[2].plot(iso_dates, qs[1] / pos, color="#e6550d", lw=2, label="PPC median (scaled)")
    ax[2].plot(
        iso_dates, ctx["shared"]["obs_suspect_iso_val"], "k.-", label="Observed suspect-in-iso"
    )
    ax[2].set_title(f"Suspect-in-iso overlay (scaled 1/{pos:.2f})")
    ax[2].legend()
    ax[2].tick_params(axis="x", rotation=45)
    # 4) full-window model iso stock vs raw national susp series (the older, longer overlay)
    qt = np.percentile(iso_stock_pp, [5, 50, 95], axis=0)
    ax[3].fill_between(
        dates, qt[0], qt[2], alpha=0.25, color="#54278f", label="Model TOTAL true-iso 5-95%"
    )
    ax[3].plot(dates, qt[1], color="#54278f", lw=2, label="Model median")
    if len(ctx["iso_stock_obs"]):
        ax[3].plot(
            ctx["iso_stock_obs"]["date"],
            ctx["iso_stock_obs"]["value"],
            "k.-",
            label="Observed national suspect ward census",
        )
    ax[3].set_title("Full-window iso stock overlay (no scaling, raw series)")
    ax[3].legend()
    ax[3].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / "02_capacity_iso.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_posterior(post) -> None:
    # 14 calibrated params + 1 derived (seed_day) → 4×4 grid with one spare for legend / notes.
    n = len(CAL)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow))
    ax = ax.ravel()
    for i, k in enumerate(CAL):
        ax[i].hist(post[k], bins=40, color="#2171b5", alpha=0.75)
        ax[i].set_title(f"{k}  med {np.median(post[k]):.3f}", fontsize=9)
    for j in range(n, len(ax)):
        ax[j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "03_posterior.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
