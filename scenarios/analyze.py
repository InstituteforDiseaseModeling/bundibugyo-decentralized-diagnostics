"""Analysis: turn results.jsonl into the reported answers.

Reads only `outputs/results.jsonl` (scalars) and writes `outputs/analysis.json` plus a
readable digest to stdout. No simulation. WORLDS and DRIV are pure post-processing over
BASE/PLACE.

Reporting contract: the headline is **% averted**, a CRN-paired within-draw difference.
Absolute burden is an explicit UPPER BOUND and never a central estimate. Placement
*locations* are not actionable under this posterior.

Run:  uv run python analyze.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys as _sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bdbv import engine as E  # CAL is read from the shared engine

HERE = Path(__file__).parent
OUT = HERE / "outputs"

# Replicate: `--tag b` reads results_b/placement_b/report_b and writes analysis_b,
# matching run.py's suffix, so the two sweeps are analysed independently and compared.
SFX = ""

# 2014 West Africa anchor: 11,310 deaths ~ 28,600 infections
WA_DEATHS, WA_INF = 11_310, 28_600
# Read from config so a re-centred report_marks flows through without editing code.
try:
    MARKS = tuple(
        yaml.safe_load((HERE / "config.yaml").read_text())["intervention"]["report_marks"]
    )
except (OSError, KeyError, TypeError, yaml.YAMLError):
    MARKS = (30, 90, 180, 540)


def pctl(x, q=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{i}": None for i in q}
    return {f"p{i}": round(float(np.percentile(x, i)), 4) for i in q}


def load():
    """Stream results.jsonl. MAPS rows are reduced to running per-zone sums rather than
    retained: 20,000 rows x 3 x 470 values is ~28M numbers (~2 GB as Python objects) and
    only the sums are ever used."""
    rows = defaultdict(list)
    path = OUT / f"results{SFX}.jsonl"
    n_bad = 0
    macc = {"n": 0, "sq": None, "av": None, "avd": None, "Z": 0}
    with path.open() as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            st = r.get("stream", "?")
            if st == "maps":
                if r.get("n", 0) <= 0:
                    continue
                sq = np.asarray(r["sq_by_zone"], float)
                if macc["sq"] is None:
                    macc.update(
                        Z=sq.size,
                        sq=np.zeros(sq.size),
                        av=np.zeros(sq.size),
                        avd=np.zeros(sq.size),
                    )
                macc["sq"] += sq
                macc["av"] += np.asarray(r["av_by_zone"], float)
                macc["avd"] += np.asarray(r["avd_by_zone"], float)
                macc["n"] += 1
                continue
            rows[st].append(r)
    if n_bad:
        print(f"[warn] {n_bad} unparseable lines (likely a truncated final write)")
    rows["_maps_acc"] = macc
    return rows


# ---------------------------------------------------------------------------


def q_base(base):
    live = [r for r in base if r.get("n", 0) > 0]
    burn = len(base) - len(live)
    out = {
        "n_rows": len(base),
        "n_burnout": burn,
        "hit_cap_frac": round(float(np.mean([bool(r.get("hit_cap")) for r in live])), 4),
    }
    for m in MARKS:
        n = [r[f"n_{m}"] for r in live if f"n_{m}" in r]
        d = [r[f"d_{m}"] for r in live if f"d_{m}" in r]
        z = [r[f"z_{m}"] for r in live if f"z_{m}" in r]
        out[f"inf_{m}"] = pctl(n)
        out[f"deaths_{m}"] = pctl(d)
        out[f"zones_{m}"] = pctl(z)
        out[f"vs_WA_deaths_{m}"] = pctl(np.array(d, float) / WA_DEATHS)
    out["n_at_cutoff"] = pctl([r["n_at_cutoff"] for r in live])
    out["zones_at_cutoff"] = pctl([r["zones_at_cutoff"] for r in live])
    return out


def q_worlds(base):
    """Variance decomposition + burden grouping. K continuations per theta make the
    split direct: within-theta = realization luck, between-theta = parametric."""
    live = [r for r in base if r.get("n", 0) > 0 and "d_540" in r]
    by_i = defaultdict(list)
    for r in live:
        by_i[r["i"]].append(r["d_540"])
    paired = {i: v for i, v in by_i.items() if len(v) >= 2}
    out = {"n_theta": len(by_i), "n_theta_with_K": len(paired)}
    if paired:
        K = float(np.mean([len(v) for v in paired.values()]))
        within = float(np.mean([np.var(v, ddof=1) for v in paired.values()]))
        var_of_means = float(np.var([np.mean(v) for v in paired.values()], ddof=1))
        # Var(mean of K) = between + within/K  ->  unbiased between
        between = max(var_of_means - within / K, 0.0)
        tot = within + between
        out.update(
            K_mean=round(K, 2),
            within_theta_var=within,
            between_theta_var=between,
            stochastic_share=round(within / tot, 4) if tot else None,
            parametric_share=round(between / tot, 4) if tot else None,
            note=(
                "between-theta is bias-corrected as Var(mean_K) - within/K; "
                "if it clamps at 0 the sample cannot separate the two"
            ),
        )
    # burden groups: KDE valleys on log10 deaths (theta-means, so luck is averaged out)
    dm = np.array([np.mean(v) for v in by_i.values()], float)
    dm = dm[dm > 0]
    if dm.size > 20:
        from scipy.stats import gaussian_kde

        lg = np.log10(dm)
        grid = np.linspace(lg.min(), lg.max(), 512)
        dens = gaussian_kde(lg)(grid)
        # interior local minima = group boundaries
        loc = [
            j for j in range(1, len(grid) - 1) if dens[j] < dens[j - 1] and dens[j] <= dens[j + 1]
        ]
        cand = [
            float(10 ** grid[j]) for j in loc if dens[j] < 0.6 * dens.max()
        ]  # ignore shallow ripples
        # A "burden world" must hold real posterior mass. Without this, tail ripples
        # get promoted to group boundaries and the grouping looks structured when it
        # is not: the raw cuts carve off slivers of a fraction of a percent each and
        # leave almost everything in one lump. Merge anything below MIN_MASS into its neighbour.
        MIN_MASS = 0.02
        cuts = []
        for c in sorted(cand):
            lo = cuts[-1] if cuts else 0.0
            if float(((dm >= lo) & (dm < c)).mean()) >= MIN_MASS:
                cuts.append(c)
        if cuts and float((dm >= cuts[-1]).mean()) < MIN_MASS:
            cuts.pop()
        out["burden_group_cuts_deaths"] = [round(c, 1) for c in cuts]
        out["burden_group_cuts_rejected"] = [round(c, 1) for c in cand if c not in cuts]
        edges = [0.0] + cuts + [np.inf]
        groups = []
        for a, b in itertools.pairwise(edges):
            sel = dm[(dm >= a) & (dm < b)]
            if sel.size:
                groups.append(
                    {
                        "lo": round(a, 1),
                        "hi": (None if b == np.inf else round(b, 1)),
                        "n": int(sel.size),
                        "frac": round(sel.size / dm.size, 4),
                        "median_deaths": round(float(np.median(sel)), 1),
                        "vs_WA": round(float(np.median(sel)) / WA_DEATHS, 2),
                    }
                )
        out["burden_groups"] = groups
        out["n_distinct_worlds"] = len(groups)
        if len(groups) == 1:
            out["single_world_note"] = (
                "ONE burden world, not several. Every posterior draw saturates, so there is "
                "no multi-modal burden structure to group -- a direct consequence of the "
                "forward operator, which does not permit small outbreaks."
            )
    return out


def q_future(base):
    """Split every burden series into ALREADY LOCKED IN vs STILL TO COME at the cutoff.

    % of total burden is the wrong metric when most of the epidemic precedes the decision:
    it deflates every arm ~4x for a reason unrelated to intervention quality. That applies
    to the BURDEN figures too, not just the averted ones: a histogram of cumulative total invites reading the whole distribution
    as in-play, when most of it is already determined. We cannot change the past.
    """
    live = [r for r in base if r.get("n", 0) > 0]
    by = defaultdict(lambda: defaultdict(list))
    pairs = (
        ("c_540", "c_at_cutoff", "confirmed_cases"),
        ("cd_540", "cd_at_cutoff", "confirmed_deaths"),
        ("n_540", "n_at_cutoff", "latent_infections"),
        ("d_540", "d_at_cutoff", "latent_deaths"),
    )
    for r in live:
        for tot, cut, _ in pairs:
            if tot in r and cut in r:
                by[tot][r["i"]].append(r[tot])
                by[cut][r["i"]].append(r[cut])
    out = {}
    for tot, cut, name in pairs:
        if tot not in by:
            continue
        T = np.array([np.mean(v) for v in by[tot].values()], float)
        C = np.array([np.mean(v) for v in by[cut].values()], float)
        F = T - C
        out[name] = {
            "total": pctl(T),
            "locked_in": pctl(C),
            "still_to_come": pctl(F),
            "future_share_of_total": round(float(np.median(F / np.maximum(T, 1))), 4),
        }
    return out


def _pct_averted(rows, key="av_total"):
    return np.array([r[key] / r["n"] for r in rows if r.get("n", 0) > 0], float)


def q_time(time_rows):
    out = {}
    for mode in sorted({r["mode"] for r in time_rows}):
        for bud in sorted({r["budget"] for r in time_rows}):
            key = f"{mode}_b{bud}"
            per = {}
            for dl in sorted({r["delay"] for r in time_rows}):
                sel = [
                    r
                    for r in time_rows
                    if r["mode"] == mode and r["budget"] == bud and r["delay"] == dl
                ]
                if sel:
                    per[dl] = pctl(100 * _pct_averted(sel))
            out[key] = per
            # cost of waiting, relative to same-day deployment
            if 0 in per and per[0]["p50"]:
                base = per[0]["p50"]
                out[key + "_rel_to_day0"] = {
                    dl: round(v["p50"] / base, 4) for dl, v in per.items() if v["p50"]
                }
    return out


def q_ramp(ramp_rows):
    """The national-clock contrast.

    Decision rule 2: no averted figure travels without the ramp it was computed under, and
    the ramp's own contribution must never be folded into the diagnostics'. So each arm
    reports its OWN forward burden and its OWN within-arm averted figure. Cross-arm
    differences are burden comparisons only — the averted columns are never differenced
    across arms, because continuations desync after day 229 (see run.py's RAMP note).
    """
    out = {}
    for arm in sorted({r["arm"] for r in ramp_rows}):
        sel = [r for r in ramp_rows if r["arm"] == arm]
        live = [r for r in sel if r.get("n", 0) > 0]
        if not live:
            continue
        k0 = [r for r in live if r.get("k") == 0 and "avd_total" in r]
        rec = {
            "role": sel[0].get("role"),
            "nat_scale_T": sel[0].get("nat_scale_T"),
            "nat_scale_min": sel[0].get("nat_scale_min"),
            "n_rows": len(sel),
            "burnout_frac": round(1 - len(live) / len(sel), 4),
            "hit_cap_frac": round(float(np.mean([bool(r.get("hit_cap")) for r in live])), 4),
            "latent_infections": pctl(np.array([r["n"] for r in live], float)),
            "latent_deaths": pctl(np.array([r["deaths"] for r in live], float)),
            "deaths_post_cutoff": pctl(np.array([r["d_post_cut"] for r in live], float)),
            "zones_invaded": pctl(np.array([r["zones"] for r in live], float)),
            "last_infection_days": pctl(np.array([r["last_inf"] for r in live], float)),
        }
        if k0:
            avd = np.array([r["avd_total"] for r in k0], float)
            dpc = np.array([r["d_post_cut"] for r in k0], float)
            rec["rep_arm_deaths_averted"] = pctl(avd)
            with np.errstate(divide="ignore", invalid="ignore"):
                rec["rep_arm_pct_of_postcut_deaths"] = pctl(
                    100 * np.divide(avd, dpc, out=np.full_like(avd, np.nan), where=dpc > 0)
                )
        out[arm] = rec
    return out


def q_place(place_rows):
    out = {}
    for mode in sorted({r["mode"] for r in place_rows}):
        for bud in sorted({r["budget"] for r in place_rows}):
            g = [
                r
                for r in place_rows
                if r["strategy"] == "greedy" and r["mode"] == mode and r["budget"] == bud
            ]
            rnd = [
                r
                for r in place_rows
                if r["strategy"] == "random" and r["mode"] == mode and r["budget"] == bud
            ]
            if not g or not rnd:
                continue
            gp, rp = 100 * _pct_averted(g), 100 * _pct_averted(rnd)
            # paired gap: greedy minus that draw's own mean random
            rmean = defaultdict(list)
            for r in rnd:
                if r.get("n", 0) > 0:
                    rmean[r["i"]].append(r["av_total"] / r["n"])
            gap = np.array(
                [
                    100 * (r["av_total"] / r["n"] - np.mean(rmean[r["i"]]))
                    for r in g
                    if r.get("n", 0) > 0 and r["i"] in rmean
                ],
                float,
            )
            out[f"{mode}_b{bud}"] = {
                "greedy": pctl(gp),
                "random": pctl(rp),
                "paired_gap_pp": pctl(gap),
                "greedy_beats_random_frac": round(float((gap > 0).mean()), 4),
            }
    return out


def q_pm(pm_rows):
    """PM arms hold the representative MACHINE placement fixed and vary SDB on top, so
    the marginal SDB effect is the arm minus that draw's own scope='none' row."""
    base = {
        r["i"]: r["av_total"] / r["n"]
        for r in pm_rows
        if r["scope"] == "none" and r.get("sdb_target") is None and r.get("n", 0) > 0
    }
    out = {
        "sq_coverage": pctl([r["sq_cov"] for r in pm_rows if "sq_cov" in r]),
        "machine_only_pct_averted": pctl(100 * np.array(list(base.values()), float)),
    }
    for scope in sorted({r["scope"] for r in pm_rows}):
        for tgt in sorted({r.get("sdb_target") for r in pm_rows}, key=lambda t: (t is None, t)):
            sel = [
                r
                for r in pm_rows
                if r["scope"] == scope
                and r.get("sdb_target") == tgt
                and r.get("n", 0) > 0
                and r["i"] in base
            ]
            if not sel:
                continue
            tot = 100 * _pct_averted(sel)
            marg = np.array([100 * (r["av_total"] / r["n"] - base[r["i"]]) for r in sel], float)
            out[f"{scope}_{tgt}"] = {
                "total_pct_averted": pctl(tot),
                "marginal_vs_machine_only_pp": pctl(marg),
                "eff_coverage": pctl([r["sdb_eff"] for r in sel]),
            }
    return out


def q_casc(casc_rows):
    """Leave-one-out: contribution of a channel = full minus (full without it)."""
    by = defaultdict(dict)
    for r in casc_rows:
        if r.get("n", 0) > 0:
            by[r["i"]][r["channel"]] = r["av_total"] / r["n"]
    out = {}
    full = np.array([v["full"] for v in by.values() if "full" in v], float)
    out["full_pct_averted"] = pctl(100 * full)
    for ch in ("pcr", "turnaround", "sdb"):
        contrib = np.array(
            [
                100 * (v["full"] - v[f"no_{ch}"])
                for v in by.values()
                if "full" in v and f"no_{ch}" in v
            ],
            float,
        )
        out[f"contrib_{ch}_pp"] = pctl(contrib)
    # share of the full effect attributable to each channel (medians, may not sum to 1)
    med = {ch: out[f"contrib_{ch}_pp"]["p50"] for ch in ("pcr", "turnaround", "sdb")}
    s = sum(v for v in med.values() if v)
    if s:
        out["channel_share_of_LOO_total"] = {k: round(v / s, 4) for k, v in med.items() if v}
    out["note"] = (
        "leave-one-out contributions need not sum to the full effect; the "
        "channels interact (a machine both raises PCR and cuts turnaround)"
    )
    return out


def q_maps(macc, zones):
    n = macc["n"]
    if not n or macc["sq"] is None:
        return {}
    sq = macc["sq"].copy()
    av = macc["av"].copy()
    avd = macc["avd"].copy()
    sq /= n
    av /= n
    avd /= n
    resid = sq - av
    frac = np.divide(av, sq, out=np.zeros_like(av), where=sq > 0)
    ordr = np.argsort(av)[::-1]
    return {
        "n_maps_rows": n,
        # FULL per-zone means, so the choropleth colours all Z zones rather than a top-20
        "by_zone": {
            "zone": list(zones),
            "sq": [round(float(v), 2) for v in sq],
            "averted": [round(float(v), 2) for v in av],
            "residual": [round(float(v), 2) for v in resid],
            "deaths_averted": [round(float(v), 2) for v in avd],
            "pct_averted": [round(100 * float(v), 2) for v in frac],
        },
        "total_sq": round(float(sq.sum()), 1),
        "total_averted": round(float(av.sum()), 1),
        "overall_pct_averted": round(100 * float(av.sum() / max(sq.sum(), 1)), 2),
        "top20_averted": [
            {
                "zone": zones[z],
                "sq": round(float(sq[z]), 1),
                "averted": round(float(av[z]), 1),
                "pct": round(100 * float(frac[z]), 1),
            }
            for z in ordr[:20]
        ],
        "top20_residual": [
            {
                "zone": zones[z],
                "residual": round(float(resid[z]), 1),
                "pct_averted": round(100 * float(frac[z]), 1),
            }
            for z in np.argsort(resid)[::-1][:20]
        ],
        "deaths_averted_total": round(float(avd.sum()), 1),
    }


def q_driv(base, place_rows):
    """Which posterior parameters drive averted burden? Spearman on the representative
    greedy arm, plus the same against baseline size for contrast."""
    from scipy.stats import spearmanr

    th = {}
    for r in base:
        if r.get("k", 0) == 0 and r.get("n", 0) > 0:
            th[r["i"]] = r
    # The driver set is EVERY calibrated parameter, read off the rows themselves, which
    # `run.py` populated from the shared engine's CAL. A hand-listed set risks including
    # names that are FIXED at a constant here (`nat_half`, `nat_floor`), which would give a
    # Spearman against a constant column without erroring. `R_floor` — this posterior's
    # forward-transmission controller, and what the brake stratification keys on — enters
    # by the same mechanism, without anyone remembering it.
    keys = tuple(k for k in E.CAL if k in th[next(iter(th))])
    rep = [
        r
        for r in place_rows
        if r["strategy"] == "greedy"
        and r["mode"] == "local"
        and r.get("n", 0) > 0
        and r["i"] in th
    ]
    if not rep:
        return {}
    buds = sorted({r["budget"] for r in rep})
    bud = buds[len(buds) // 2]
    rep = [r for r in rep if r["budget"] == bud]
    y = np.array([r["av_total"] / r["n"] for r in rep], float)
    size = np.array([th[r["i"]]["n_540"] for r in rep], float)
    out = {"budget_used": int(bud), "n": len(rep)}
    for k in keys:
        x = np.array([th[r["i"]][k] for r in rep], float)
        rho_a, p_a = spearmanr(x, y)
        rho_s, _ = spearmanr(x, size)
        out[k] = {
            "rho_vs_pct_averted": round(float(rho_a), 4),
            "p": round(float(p_a), 6),
            "rho_vs_baseline_size": round(float(rho_s), 4),
        }
    rho, _ = spearmanr(size, y)
    out["baseline_size_vs_pct_averted"] = round(float(rho), 4)
    return out


def q_avertable(base, place_rows, pm_rows, time_rows, casc_rows):
    """Absolute deaths averted, and averted burden as a fraction of what is still
    AVERTABLE — rather than of total epidemic burden.

    Roughly three quarters of the epidemic is already in the past at the
    cutoff, so a fraction of TOTAL burden understates the intervention by ~4x for a
    reason that has nothing to do with how good the intervention is. Two honest
    denominators:
      post-cutoff    what is still in the future when the DECISION is made (today)
      post-deploy    what is still in the future when the machines actually LAND
    The second is always the smaller pool and the fairer test of the hardware itself;
    the first is the one a decision-maker at the cutoff actually faces.
    """
    out = {}
    live = [r for r in base if r.get("k", 0) == 0 and r.get("n", 0) > 0 and "n_post_cut" in r]
    if live:
        out["baseline"] = {
            "total_infections": pctl([r["n"] for r in live]),
            "total_deaths": pctl([r["deaths"] for r in live]),
            "deaths_already_at_cutoff": pctl([r["d_at_cutoff"] for r in live]),
            "deaths_still_avertable": pctl([r["d_post_cut"] for r in live]),
            "infections_still_avertable": pctl([r["n_post_cut"] for r in live]),
            "frac_of_epidemic_already_past": pctl([r["n_at_cutoff"] / r["n"] for r in live]),
        }

    def block(rows_, filt, label):
        sel = [r for r in rows_ if filt(r) and r.get("n", 0) > 0 and r.get("d_post_cut")]
        if not sel:
            return
        out[label] = {
            "n": len(sel),
            "deaths_averted_ABSOLUTE": pctl([r["avd_total"] for r in sel]),
            "infections_averted_ABSOLUTE": pctl([r["av_total"] for r in sel]),
            "pct_of_total_deaths": pctl(
                [100 * r["avd_total"] / max(r.get("deaths") or 1, 1) for r in sel]
            )
            if all("deaths" in r for r in sel)
            else None,
            "pct_of_AVERTABLE_deaths_postcut": pctl(
                [100 * r["avd_total"] / r["d_post_cut"] for r in sel if r["d_post_cut"] > 0]
            ),
            "pct_of_AVERTABLE_deaths_postdeploy": pctl(
                [100 * r["avd_total"] / r["d_post_dep"] for r in sel if r.get("d_post_dep", 0) > 0]
            ),
            "pct_of_AVERTABLE_infections_postcut": pctl(
                [100 * r["av_total"] / r["n_post_cut"] for r in sel if r["n_post_cut"] > 0]
            ),
        }
        out[label] = {k2: v2 for k2, v2 in out[label].items() if v2 is not None}

    buds = sorted({r["budget"] for r in place_rows}) if place_rows else []
    for b in buds:
        block(
            place_rows,
            lambda r, b=b: r["strategy"] == "greedy" and r["mode"] == "local" and r["budget"] == b,
            f"machines_only_greedy_local_b{b}",
        )
    block(
        pm_rows,
        lambda r: r["scope"] == "all" and r.get("sdb_target") == 0.5,
        "machines_plus_burial_50",
    )
    block(
        pm_rows,
        lambda r: r["scope"] == "all" and r.get("sdb_target") == 0.75,
        "machines_plus_burial_75",
    )
    if time_rows:
        # Hoist the max budget OUT of the predicate. Computing it inside the lambda
        # rebuilt a set over all ~192k time rows for every row tested -- O(n^2), and it
        # turned this function into a 30-minute CPU burn.
        bmax = max(r["budget"] for r in time_rows)
        for dl in sorted({r["delay"] for r in time_rows}):
            block(
                time_rows,
                lambda r, dl=dl, bmax=bmax: (
                    r["mode"] == "local" and r["delay"] == dl and r["budget"] == bmax
                ),
                f"delay_{dl}d_machines_only",
            )
    return out


BRAKE_KEYS = {
    # `nat_half` / `nat_floor` are NOT calibrated under `nat_brake_placement: global` — they
    # sit at a constant 0.0 — so stratifying on them would produce four identical quartiles
    # without erroring.
    #
    # `R_floor` is their analogue and the only calibrated parameter left that sets forward
    # transmission. Under `global` placement the late-time asymptote is S_min * R_floor, so
    # a LARGER R_floor is a WEAKER effective brake. Q1 (small floor) is the strongest brake,
    # Q4 (large floor) the weakest.
    #
    # Its 90% CI is [0.703, 1.761] and the median is 1.134, i.e. it straddles 1: for the
    # upper part of that range the epidemic is terminated ONLY by the national scale factor.
    # That is why the stratified table, not the pooled figure, is the result.
    "R_floor": (
        "strongest brake",
        "weakest brake",
        (
            "r_eff -> S_nat * R_floor once B_loc ~ 0; larger R_floor = higher forward "
            "asymptote = weaker brake = more epidemic left to avert"
        ),
    ),
}


def q_brake(base, place_rows, pm_rows, key="R_floor"):
    """% averted stratified by the forward-transmission parameter.

    Reporting a single pooled % averted hides that the answer is essentially a function of
    how close forward R_eff sits to 1: brake quartiles spread machines-only averted over a
    nearly 8x monotone range on that one parameter.

    Under this operator that parameter is `R_floor` (see BRAKE_KEYS). It is CALIBRATED, so
    it shows up in the driver analysis rather than having to be inferred.
    """
    lo_lbl, hi_lbl, note = BRAKE_KEYS[key]
    th = {r["i"]: r for r in base if r.get("k", 0) == 0 and r.get("n", 0) > 0 and key in r}
    if not th:
        return {}
    nh = np.array([r[key] for r in th.values()], float)
    qs = np.percentile(nh, [25, 50, 75])
    out = {key: pctl(nh), "quartile_edges": [round(float(q), 4) for q in qs], "note": note}

    def strat(rows_, label, filt):
        sel = [r for r in rows_ if filt(r) and r.get("n", 0) > 0 and r["i"] in th]
        if not sel:
            return
        bins = {
            f"Q1 ({lo_lbl})": lambda v: v <= qs[0],
            "Q2": lambda v: qs[0] < v <= qs[1],
            "Q3": lambda v: qs[1] < v <= qs[2],
            f"Q4 ({hi_lbl})": lambda v: v > qs[2],
        }
        d = {}
        for bn, bf in bins.items():
            vals = [100 * r["av_total"] / r["n"] for r in sel if bf(th[r["i"]][key])]
            sizes = [th[r["i"]]["n_540"] for r in sel if bf(th[r["i"]][key])]
            if vals:
                d[bn] = {
                    "n": len(vals),
                    "pct_averted": pctl(vals),
                    "median_baseline": round(float(np.median(sizes)), 0),
                }
        out[label] = d

    rep_b = sorted({r["budget"] for r in place_rows})
    rep_b = rep_b[len(rep_b) // 2] if rep_b else None
    strat(
        place_rows,
        f"place_greedy_local_b{rep_b}",
        lambda r: r["strategy"] == "greedy" and r["mode"] == "local" and r["budget"] == rep_b,
    )
    strat(pm_rows, "pm_all_0.75", lambda r: r["scope"] == "all" and r.get("sdb_target") == 0.75)
    return out


def main():
    rows = load()
    print(f"[load] streams: { {k: len(v) for k, v in sorted(rows.items())} }")
    placement = json.loads((OUT / f"placement{SFX}.json").read_text())
    zones = placement.get("greedy_zones_all") or None
    _rp = OUT / f"report{SFX}.json"
    rep = json.loads(_rp.read_text()) if _rp.exists() else {}

    res = {"report": rep, "anchor": {"WA_deaths": WA_DEATHS, "WA_infections": WA_INF}}
    # Observed-to-date, read from the shared engine so it cannot drift from the data pin.
    # These are the "TODAY" markers in figure 02 and the only hard facts on that chart.
    try:
        _ctx = E.build_context(yaml.safe_load((HERE / "config.yaml").read_text()))
        res["observed_at_cutoff"] = {
            "confirmed_cases": float(_ctx["obs_nat_cum"][-1]),
            "confirmed_deaths": float(_ctx["shared"].get("obs_deaths_cum_end", float("nan"))),
            "date": "2026-08-02",
        }
    except (OSError, KeyError, ValueError) as _e:
        print(f"[warn] could not read observed-at-cutoff: {_e}")
    res["BASE"] = q_base(rows["base"])
    res["FUTURE"] = q_future(rows["base"])
    res["WORLDS"] = q_worlds(rows["base"])
    if rows["time"]:
        res["TIME"] = q_time(rows["time"])
    if rows["place"]:
        res["PLACE"] = q_place(rows["place"])
    if rows["ramp"]:
        res["RAMP"] = q_ramp(rows["ramp"])
    if rows["pm"]:
        res["PM"] = q_pm(rows["pm"])
    if rows["casc"]:
        res["CASC"] = q_casc(rows["casc"])
    if rows["_maps_acc"]["n"]:
        zl = zones or [f"z{j}" for j in range(rows["_maps_acc"]["Z"])]
        res["MAPS"] = q_maps(rows["_maps_acc"], zl)
    if rows["base"] and rows["place"]:
        res["DRIV"] = q_driv(rows["base"], rows["place"])
        # ONE brake section: `nat_half` / `nat_floor` are not calibrated here, so `R_floor`
        # is the whole story. Driven off BRAKE_KEYS so the section list cannot drift from
        # the stratification keys.
        for _bk in BRAKE_KEYS:
            res[f"BRAKE_{_bk}"] = q_brake(rows["base"], rows["place"], rows["pm"], key=_bk)
        res["AVERTABLE"] = q_avertable(
            rows["base"], rows["place"], rows["pm"], rows["time"], rows["casc"]
        )
    res["greedy_top20"] = placement["greedy_zones"][:20]

    (OUT / f"analysis{SFX}.json").write_text(json.dumps(res, indent=1, default=str))
    _digest(res)
    print(f"\n[done] -> {OUT / f'analysis{SFX}.json'}")


def _digest(res):
    b = res["BASE"]
    print("\n" + "=" * 74)
    print("BASE — forward baseline (ABSOLUTE = UPPER BOUND, caveat 1)")
    print("=" * 74)
    print(f"  rows {b['n_rows']}  burnouts {b['n_burnout']}  hit_cap {b['hit_cap_frac']:.1%}")
    for m in MARKS:
        i, d, z = b[f"inf_{m}"], b[f"deaths_{m}"], b[f"zones_{m}"]
        wa = b[f"vs_WA_deaths_{m}"]
        print(
            f"  +{m:3d}d  inf {i['p50']:>10,.0f} [{i['p5']:>9,.0f}, {i['p95']:>10,.0f}]"
            f"  deaths {d['p50']:>9,.0f}  zones {z['p50']:>4.0f}  = {wa['p50']:>6.1f}x WA"
        )
    w = res["WORLDS"]
    print(
        f"\nWORLDS — variance: stochastic {w.get('stochastic_share')} / "
        f"parametric {w.get('parametric_share')}  (K={w.get('K_mean')}, "
        f"{w.get('n_theta')} thetas)"
    )
    for g in w.get("burden_groups", []):
        print(
            f"  group deaths [{g['lo']:,.0f}, {g['hi'] if g['hi'] else 'inf'}]: "
            f"{g['frac']:.1%} of draws, median {g['median_deaths']:,.0f} = {g['vs_WA']}x WA"
        )

    if "PLACE" in res:
        print("\n" + "=" * 74)
        print("PLACE — % averted (paired within draw). Locations NOT actionable (caveat 1b)")
        print("=" * 74)
        for k, v in res["PLACE"].items():
            print(
                f"  {k:16s} greedy {v['greedy']['p50']:5.1f}%  random {v['random']['p50']:5.1f}%"
                f"  gap {v['paired_gap_pp']['p50']:+5.1f}pp"
                f"  greedy wins {v['greedy_beats_random_frac']:.0%}"
            )
    if "TIME" in res:
        print("\nTIME — % of same-day benefit retained")
        for k, v in res["TIME"].items():
            if k.endswith("_rel_to_day0"):
                print(
                    f"  {k.removesuffix('_rel_to_day0'):16s} "
                    + "  ".join(
                        f"{d}d:{r:.2f}" for d, r in sorted(v.items(), key=lambda t: int(t[0]))
                    )
                )
    if "PM" in res:
        print("\nPM — SDB coverage (marginal over machine-only; RDT is coverage-only, caveat 2)")
        print(f"  machine-only baseline: {res['PM']['machine_only_pct_averted']['p50']:.1f}%")
        for k, v in res["PM"].items():
            if isinstance(v, dict) and "marginal_vs_machine_only_pp" in v:
                print(
                    f"  {k:16s} total {v['total_pct_averted']['p50']:5.1f}%"
                    f"  marginal {v['marginal_vs_machine_only_pp']['p50']:+5.1f}pp"
                )
    if "CASC" in res:
        c = res["CASC"]
        print(f"\nCASC — full {c['full_pct_averted']['p50']:.1f}%; leave-one-out contributions")
        for ch in ("pcr", "turnaround", "sdb"):
            print(f"  {ch:12s} {c[f'contrib_{ch}_pp']['p50']:+5.1f}pp")
    if "DRIV" in res:
        print("\nDRIV — Spearman rho vs % averted (and vs baseline size)")
        for k, v in res["DRIV"].items():
            if isinstance(v, dict):
                print(
                    f"  {k:14s} {v['rho_vs_pct_averted']:+.3f} (size {v['rho_vs_baseline_size']:+.3f})"
                )
        print(
            f"  {'baseline size':14s} {res['DRIV'].get('baseline_size_vs_pct_averted'):+.3f}"
            f"   <- bigger epidemics are harder to avert PROPORTIONALLY if negative"
        )
    if res.get("AVERTABLE"):
        av = res["AVERTABLE"]
        b = av.get("baseline", {})
        print("\n" + "=" * 74)
        print("AVERTABLE — absolute deaths averted, and the RIGHT denominator")
        print("=" * 74)
        if b:
            print(f"  total deaths                 {b['total_deaths']['p50']:>8,.0f}")
            print(
                f"  already locked in at cutoff  {b['deaths_already_at_cutoff']['p50']:>8,.0f}"
                f"   ({100 * b['frac_of_epidemic_already_past']['p50']:.0f}% of epidemic is past)"
            )
            print(
                f"  STILL AVERTABLE (deaths)     {b['deaths_still_avertable']['p50']:>8,.0f}"
                f"   [{b['deaths_still_avertable']['p5']:,.0f}, {b['deaths_still_avertable']['p95']:,.0f}]"
            )
        print(f"\n  {'arm':34s} {'deaths averted':>22s}  {'% of avertable':>14s}")
        for k, v in av.items():
            if k == "baseline" or not isinstance(v, dict):
                continue
            d = v.get("deaths_averted_ABSOLUTE", {})
            pc = v.get("pct_of_AVERTABLE_deaths_postcut", {})
            if d.get("p50") is None:
                continue
            print(
                f"  {k:34s} {d['p50']:>8,.0f} [{d['p5']:>5,.0f}, {d['p95']:>6,.0f}]"
                f"  {pc.get('p50', 0):>10.1f}%"
            )
    for sect, pkey, fmt in ((f"BRAKE_{_k}", _k, ".3f") for _k in BRAKE_KEYS):
        b = res.get(sect)
        if not b:
            continue
        print(f"\n{sect} — % averted stratified by {pkey}")
        print(f"  {pkey} p50 {b[pkey]['p50']:{fmt}}  quartile edges {b['quartile_edges']}")
        for k, v in b.items():
            if isinstance(v, dict) and any(kk.startswith("Q") for kk in v):
                print(f"  {k}:")
                for bn, bv in v.items():
                    print(
                        f"     {bn:24s} n={bv['n']:4d}  averted {bv['pct_averted']['p50']:5.2f}%"
                        f"  [{bv['pct_averted']['p5']:.2f}, {bv['pct_averted']['p95']:.2f}]"
                        f"  baseline {bv['median_baseline']:,.0f}"
                    )
    print(f"\ngreedy top-10: {res['greedy_top20'][:10]}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--tag", type=str, default="", help="artefact suffix, matches run.py --tag")
    _a = _ap.parse_args()
    SFX = f"_{_a.tag}" if _a.tag else ""
    main()
