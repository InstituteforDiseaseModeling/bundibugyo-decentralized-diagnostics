"""The intervention sweep over the calibrated posterior, under a chosen national ramp.

One task per (draw, continuation): build the forward tree once, run every arm that needs
it, stream scalars to JSONL, discard the tree. Trees are never resident as a set.

The historical portion of every tree is pinned to the stored (theta, sim_seed) — which is
keeps every forward run anchored to a scored trajectory — and only the post-cutoff
continuation branches on fresh RNG.
Continuation 0 IS the stored seed, so the K=1 streams read continuation 0.

The forward brake is NOT fitted. It is the CHOSEN `nat_scale` ramp, of which exactly ONE
ships here (arm `t090s25n2`, selected post-hoc on the forward criterion) — and R_floor,
calibrated and above 1, means the status quo cannot end without that ramp. Every number
this produces is therefore MARGINAL TO AN ASSUMED NATIONAL RESPONSE, and must be quoted
with the arm it was computed under.

The arm is never typed into logic: the code resolves it as the `ramp_arms` entry with
`role: primary`, so a one-arm config needs no code change. The arm name in this docstring
is documentation and can go stale; `outputs/report.json` records what actually ran.

Streams:
  BASE  forward baseline, accrual at the report marks             all K
  TIME  deployment delay                                          K=1 (continuation 0)
  PLACE greedy vs random x budget x mode                          K=1
  PM    safe-and-dignified-burial coverage x scope                K=1
  CASC  mechanism decomposition at the representative arm         K=1
  MAPS  per-zone averted + residual at the representative arm     all K
  RAMP  the national-clock contrast                               all K
WORLDS and DRIV are pure post-processing over BASE/PLACE output — no simulation.

Run:  uv run python run.py [--draws N] [--workers W] [--streams base,time,...] [--fresh]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import lib
from draw_split import split_greedy_and_score_draws

from bdbv import engine as E
from bdbv.model.tree import prune, tree_to_frame

OUT = HERE / "outputs"
# `ramp` is the national-clock contrast. It exists because T_nat and S_min are CONSTANTS
# in every draw, so the driver analysis is otherwise structurally blind to the single
# largest thing setting forward burden.
ALL_STREAMS = ("base", "time", "place", "pm", "casc", "maps", "ramp")
RESUME_KEYS = (
    "n_draws",
    "n_continuations",
    "split_seed",
    "score_subsample_rows",
)

# TIMERDT is TIME crossed with the RDT/burial ladder, and is NOT in ALL_STREAMS — it only runs when named explicitly, so no default invocation
# grows a stream it did not ask for.
#
# WHY it has to exist at all: the `time` stream calls scen(mask, mode) and takes
# build_scenario's defaults sdb_scope="none" / sdb_target=None, so every delay curve in
# analysis.json's TIME section is MACHINES-ONLY. The RDT channel appears only in `pm`, and
# `pm` is priced at the single representative delay. The delay x RDT cross is therefore
# not recoverable from the existing rows at all — it was never run.
#
# The ladder deliberately carries both scopes, because they are different interventions
# and the figure means different things under each:
#   machine  RDTs ride along with the labs — coverage rises only where a machine lands,
#            so the arm scales with the budget and is the "one package" story.
#   all      a national RDT/burial programme independent of where labs go — coverage
#            rises everywhere, so most of its effect is NOT attributable to placement.
# `none` is carried as the in-run null so the RDT arms are read against a curve produced
# by the same tasks, not against the older TIME section.
TIMERDT_SDB = (("none", None), ("machine", 0.75), ("all", 0.50), ("all", 0.75))
# The three fields an arm may vary. `nat_scale_t0` is deliberately NOT here: it is pinned at
# 88 so phase 1 stays bit-identical across arms; it is fixed, never tuned.
RAMP_KEYS = ("nat_scale_T", "nat_scale_min", "nat_scale_n")


def _cont_seed(sim_seed: int, k: int) -> int:
    """Continuation-k RNG seed. MUST match lib.build_tree's derivation exactly, or the RAMP
    arms would branch on a different stream than BASE and the two would not be comparable."""
    if not k:
        return int(sim_seed)
    return int(sim_seed) ^ (0x9E3779B97F4A7C15 * (k + 1) & ((1 << 63) - 1))


_S: dict = {}


def _init(shared):
    _S.clear()
    _S.update(shared)
    E.init_worker(shared["_engine"])


def _resume_metadata(
    n_draws: int,
    n_continuations: int,
    split_seed: int,
    score_rows: np.ndarray,
) -> dict:
    return {
        "n_draws": int(n_draws),
        "n_continuations": int(n_continuations),
        "split_seed": int(split_seed),
        "score_subsample_rows": [int(i) for i in score_rows],
    }


def _check_resume_metadata(results_path: Path, report_path: Path, expected: dict) -> None:
    if not results_path.exists():
        return

    observed = None
    with results_path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stream") == "_metadata":
                observed = row
                break

    if observed is None and report_path.exists():
        observed = json.loads(report_path.read_text())
    if observed is None:
        raise RuntimeError(
            f"{results_path} has no split metadata; rerun with --fresh so held-out "
            "posterior rows cannot be mixed across scenario runs"
        )

    mismatched = [key for key in RESUME_KEYS if observed.get(key) != expected[key]]
    if mismatched:
        raise RuntimeError(
            f"{results_path} was produced with different split metadata "
            f"({', '.join(mismatched)}); rerun with --fresh"
        )


# ---------------------------------------------------------------------------
# Per-(draw, continuation) worker
# ---------------------------------------------------------------------------


def _work(task):
    i, kc = task
    iv, ctx, post = _S["iv"], _S["ctx"], _S["post"]
    streams, mach = _S["streams"], _S["mach"]
    marks = ctx["marks"]

    # Two-phase: phase 1 runs the data-conditioned history to the cutoff with the STORED
    # seed (bit-identical to the calibration path, ~0.02 s so recomputing it per
    # continuation is free), then phase 2 resumes from the cutoff frontier with a fresh
    # stream. Continuation 0 keeps the stored seed and is the canonical realisation.
    T, prm, p = lib.build_tree(
        post, i, _S["_engine"], ctx["horizon_day"], cutoff_day=ctx["cutoff_day"], k=kc
    )
    n = int(T.get("n", 0))
    if n == 0:
        return [{"stream": "base", "i": int(i), "k": kc, "n": 0, "burnout": True}]

    t_inf, zi, died = T["t_infect"], T["zone_idx"], T["died"]
    rows = []
    rep = iv["representative_arm"]
    dep_rep = ctx["cutoff_day"] + int(rep["delay_days"])

    # ---- BASE (all continuations) -----------------------------------------
    if "base" in streams:
        b = {
            "stream": "base",
            "i": int(i),
            "k": kc,
            "n": n,
            "hit_cap": bool(T["hit_cap"]),
            "deaths": int(died.sum()),
            "n_at_cutoff": int((t_inf <= ctx["cutoff_day"]).sum()),
            "d_at_cutoff": int(died[t_inf <= ctx["cutoff_day"]].sum()),
            "n_post_cut": int((t_inf > ctx["cutoff_day"]).sum()),
            "d_post_cut": int(died[t_inf > ctx["cutoff_day"]].sum()),
            "zones_at_cutoff": int(np.unique(zi[t_inf <= ctx["cutoff_day"]]).size),
        }
        # ---- the driver set is DERIVED from CAL, never hand-listed --------------------
        # A hand-written parameter list is a silent-wrong-answer risk, not a crash risk:
        # names that have since moved from CAL to FIXED (`nat_half`, `nat_floor`) still
        # resolve — the engine pins them at 0.0 in FIXED_FROM_CAL so downstream lookups
        # keep working — so a stale list records a CONSTANT column for every draw and runs
        # to completion. The driver correlation is then NaN-or-zero on that constant column
        # and the stratified brake figure shows four identical quartiles.
        #
        # Reading CAL instead also means `R_floor` — this posterior's forward-transmission
        # controller, and what the brake stratification keys on — arrives without anyone
        # remembering to add it.
        b.update({k_: float(p[k_]) for k_ in E.CAL})
        # Also record CONFIRMED counts, so the forward distribution can be compared
        # like-for-like against observed-to-date and against the 2014 West Africa anchor.
        # `n_`/`d_` are LATENT (every infection / every death, detected or not), whereas
        # the `vs_WA` ratios divide by 2014's REPORTED totals — not like-for-like.
        # A case is confirmed at t_detect; a death is confirmed once the case is both
        # detected and dead, i.e. max(t_detect, t_death).
        # Use the canonical observation layer rather than reimplementing it: t_detect is
        # the CAPACITY-QUEUED confirmation time, and the confirmation cap really does
        # bind over much of the fit window, so the raw arrival time would overstate how
        # fast cases are confirmed. tree_to_frame mirrors model.simulate, so this cannot
        # drift from the calibration path.
        _kw = lib._sim_kw(p, _S["_engine"])
        _df = tree_to_frame(T, prm, confirm_capacity=_kw["confirm_capacity"])
        det = _df["detected"].to_numpy()
        t_det = _df["t_detect"].to_numpy()
        t_dth = _df["t_death"].to_numpy()
        cm = det & died & np.isfinite(t_det) & np.isfinite(t_dth)
        t_cdeath = np.maximum(t_det[cm], t_dth[cm])
        for m, md in zip(iv["report_marks"], marks):
            sel = t_inf <= md
            b[f"n_{m}"] = int(sel.sum())
            b[f"d_{m}"] = int(died[sel].sum())
            b[f"z_{m}"] = int(np.unique(zi[sel]).size)
            b[f"c_{m}"] = int((det & np.isfinite(t_det) & (t_det <= md)).sum())
            b[f"cd_{m}"] = int((t_cdeath <= md).sum())
        b["c_at_cutoff"] = int((det & np.isfinite(t_det) & (t_det <= ctx["cutoff_day"])).sum())
        b["cd_at_cutoff"] = int((t_cdeath <= ctx["cutoff_day"]).sum())
        rows.append(b)

    # ---- RAMP: the four-arm national-clock contrast -----------------------
    # Read WITHIN-ARM ONLY. Each arm's averted figure is differenced against ITS OWN SQ,
    # never across arms: continuations desync after day 229 once r_eff differs, and a
    # frozen-theta replay loses the seed's calibration, so a cross-arm difference would
    # price the desync, not the ramp.
    #
    # The arms SHARE ONE HISTORY object. That is not an optimisation -- it makes phase-1
    # equality exact by construction rather than by floating-point luck: `nat_scale` returns
    # 1.0 for every t_infect <= 229 and the cutoff is 213, so phase 1 cannot differ, and
    # reusing the object means it provably does not. Only phase 2 is re-run per arm.
    if "ramp" in streams:
        hist, prm0, p0 = lib.build_history(post, i, _S["_engine"], ctx["cutoff_day"])
        for arm in iv["ramp_arms"]:
            prm_a = dataclasses.replace(prm0, **{k: float(arm[k]) for k in RAMP_KEYS})
            T_a = lib.continue_tree(
                hist,
                prm_a,
                p0,
                _S["_engine"],
                ctx["horizon_day"],
                _cont_seed(int(post["sim_seed"][i]), kc),
            )
            n_a = int(T_a.get("n", 0))
            if n_a == 0:
                rows.append(
                    {
                        "stream": "ramp",
                        "i": int(i),
                        "k": kc,
                        "arm": arm["name"],
                        "role": arm["role"],
                        "n": 0,
                        "burnout": True,
                    }
                )
                continue
            ti_a, died_a, zi_a = T_a["t_infect"], T_a["died"], T_a["zone_idx"]
            r = dict(
                stream="ramp",
                i=int(i),
                k=kc,
                arm=arm["name"],
                role=arm["role"],
                n=n_a,
                hit_cap=bool(T_a["hit_cap"]),
                deaths=int(died_a.sum()),
                n_post_cut=int((ti_a > ctx["cutoff_day"]).sum()),
                d_post_cut=int(died_a[ti_a > ctx["cutoff_day"]].sum()),
                zones=int(np.unique(zi_a).size),
                last_inf=float(ti_a.max() - ctx["cutoff_day"]),
                R_floor=float(p0["R_floor"]),
                **{k: float(arm[k]) for k in RAMP_KEYS},
            )
            for m, md in zip(iv["report_marks"], marks):
                sel = ti_a <= md
                r[f"n_{m}"] = int(sel.sum())
                r[f"d_{m}"] = int(died_a[sel].sum())
            # The representative arm, priced against THIS arm's own SQ (continuation 0 only,
            # so the paired difference shares this arm's random numbers).
            if kc == 0:
                sc = lib.build_scenario(
                    mach[(rep["strategy"], rep["budget"])],
                    ctx,
                    iv,
                    mode=rep["mode"],
                    sdb_scope=rep["sdb_scope"],
                    sdb_target=rep["sdb_target"],
                    params=prm_a,
                )
                rm = prune(T_a, prm_a, sc, dep_rep)["removed"]
                r["av_total"] = int(rm.sum())
                r["avd_total"] = int((rm & died_a).sum())
            rows.append(r)

    # counterfactual arms only on continuation 0 (paired differences)
    if kc != 0:
        if "maps" in streams:
            rows.append(_maps_row(i, kc, T, prm, p, ctx, iv, mach, rep, dep_rep, marks))
        return rows

    def averted(sc, dep):
        """Removed counts vs this draw's own SQ, at each reporting mark.

        Also carries the AVERTABLE denominators: burden with t_infect past the cutoff
        (what is still in the future when the decision is made) and past the deployment
        day (what is still in the future when the machines actually arrive). Reporting
        only a fraction of TOTAL burden conflates intervention quality with how late in
        the epidemic we are looking.
        """
        r = prune(T, prm, sc, dep)
        rm = r["removed"]
        post_c = t_inf > ctx["cutoff_day"]
        post_d = t_inf > dep
        out = {
            "n_post_cut": int(post_c.sum()),
            "d_post_cut": int(died[post_c].sum()),
            "n_post_dep": int(post_d.sum()),
            "d_post_dep": int(died[post_d].sum()),
        }
        for m, md in zip(iv["report_marks"], marks):
            sel = t_inf <= md
            out[f"av_{m}"] = int((rm & sel).sum())
            out[f"avd_{m}"] = int((rm & sel & died).sum())
        out["av_total"] = int(rm.sum())
        out["avd_total"] = int((rm & died).sum())
        return out, rm

    def scen(mask, mode, scope="none", target=None):
        return lib.build_scenario(
            mask, ctx, iv, mode=mode, sdb_scope=scope, sdb_target=target, params=prm
        )

    # ---- TIME: delay x mode x budget --------------------------------------
    if "time" in streams:
        for delay in iv["deployment_delays"]:
            for mode in iv["modes"]:
                for bud in iv["budgets"]:
                    a, _ = averted(
                        scen(mach[("greedy", bud)], mode), ctx["cutoff_day"] + int(delay)
                    )
                    rows.append(
                        dict(
                            stream="time",
                            i=int(i),
                            n=n,
                            delay=int(delay),
                            mode=mode,
                            budget=int(bud),
                            **a,
                        )
                    )

    # ---- TIMERDT: delay x mode x budget x RDT/burial arm -------------------
    if "timerdt" in streams:
        for delay in iv["deployment_delays"]:
            for mode in iv["modes"]:
                for bud in iv["budgets"]:
                    for scope, tgt in TIMERDT_SDB:
                        sc = scen(mach[("greedy", bud)], mode, scope, tgt)
                        a, _ = averted(sc, ctx["cutoff_day"] + int(delay))
                        rows.append(
                            dict(
                                stream="timerdt",
                                i=int(i),
                                n=n,
                                delay=int(delay),
                                mode=mode,
                                budget=int(bud),
                                scope=scope,
                                sdb_target=(None if tgt is None else float(tgt)),
                                sdb_eff=float(sc["sdb_coverage"]),
                                sq_cov=float(p["cadaver_swab_coverage"]),
                                **a,
                            )
                        )

    # ---- PLACE: strategy x budget x mode ----------------------------------
    if "place" in streams:
        dep = ctx["cutoff_day"] + int(rep["delay_days"])
        for bud in iv["budgets"]:
            for mode in iv["modes"]:
                a, _ = averted(scen(mach[("greedy", bud)], mode), dep)
                rows.append(
                    dict(
                        stream="place",
                        i=int(i),
                        n=n,
                        strategy="greedy",
                        budget=int(bud),
                        mode=mode,
                        **a,
                    )
                )
                for rs in range(iv["n_random_seeds"]):
                    a, _ = averted(scen(mach[("random", bud, rs)], mode), dep)
                    rows.append(
                        dict(
                            stream="place",
                            i=int(i),
                            n=n,
                            strategy="random",
                            budget=int(bud),
                            mode=mode,
                            rand=rs,
                            **a,
                        )
                    )

    # ---- PM: SDB coverage x scope -----------------------------------------
    if "pm" in streams:
        for scope in iv["sdb_scopes"]:
            for tgt in iv["sdb_targets"]:
                sc = scen(mach[("greedy", rep["budget"])], rep["mode"], scope, tgt)
                a, _ = averted(sc, dep_rep)
                rows.append(
                    dict(
                        stream="pm",
                        i=int(i),
                        n=n,
                        scope=scope,
                        sdb_target=(None if tgt is None else float(tgt)),
                        sdb_eff=float(sc["sdb_coverage"]),
                        sq_cov=float(p["cadaver_swab_coverage"]),
                        **a,
                    )
                )

    # ---- CASC: mechanism decomposition at the representative arm ----------
    if "casc" in streams:
        full = scen(
            mach[(rep["strategy"], rep["budget"])],
            rep["mode"],
            rep["sdb_scope"],
            rep["sdb_target"],
        )
        a_full, _ = averted(full, dep_rep)
        rows.append(dict(stream="casc", i=int(i), n=n, channel="full", **a_full))
        # leave-one-out: turn each channel off and see how much averting is lost
        for ch in ("pcr", "turnaround", "sdb"):
            sc = dict(full)
            if ch == "pcr":
                sc["pcr_prob"] = np.zeros_like(full["pcr_prob"])
            elif ch == "turnaround":
                sc["turnaround"] = np.full_like(full["turnaround"], np.inf)
            else:
                sc["sdb_zones"] = np.zeros_like(full["sdb_zones"])
            a, _ = averted(sc, dep_rep)
            rows.append(dict(stream="casc", i=int(i), n=n, channel=f"no_{ch}", **a))

    if "maps" in streams:
        rows.append(_maps_row(i, kc, T, prm, p, ctx, iv, mach, rep, dep_rep, marks))
    return rows


def _maps_row(i, kc, T, prm, p, ctx, iv, mach, rep, dep_rep, marks):
    """Per-zone averted and residual burden at the representative arm."""
    sc = lib.build_scenario(
        mach[(rep["strategy"], rep["budget"])],
        ctx,
        iv,
        mode=rep["mode"],
        sdb_scope=rep["sdb_scope"],
        sdb_target=rep["sdb_target"],
        params=prm,
    )
    r = prune(T, prm, sc, dep_rep)
    rm = r["removed"]
    Z = ctx["Z"]
    return {
        "stream": "maps",
        "i": int(i),
        "k": kc,
        "n": int(T["n"]),
        "sq_by_zone": np.bincount(T["zone_idx"], minlength=Z).tolist(),
        "av_by_zone": np.bincount(T["zone_idx"][rm], minlength=Z).tolist(),
        "avd_by_zone": np.bincount(T["zone_idx"][rm & T["died"]], minlength=Z).tolist(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--conts", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--streams", type=str, default=",".join(ALL_STREAMS))
    ap.add_argument("--fresh", action="store_true")
    # Optional suffixes let independent scenario variants run without silently
    # overwriting each other's artifacts.
    ap.add_argument("--tag", type=str, default="", help="artefact suffix, e.g. 'b'")
    ap.add_argument(
        "--subsample",
        type=str,
        default=None,
        help="override intervention.subsample (per-replicate draw file)",
    )
    # Price the placement contrast at a deployment day other than the config's
    # representative 14. This overrides
    # representative_arm.delay_days IN MEMORY -- config.yaml is never edited, so the
    # override cannot be inherited by a later run that did not ask for it
    # inadvertently. It moves dep_rep, so the greedy
    # order is RE-PICKED for the later deployment, not merely re-priced; pair it with
    # --tag so the day-14 artefacts are not overwritten.
    ap.add_argument(
        "--rep-delay",
        type=int,
        default=None,
        help="override representative_arm.delay_days (days after cutoff)",
    )
    a = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    iv = cfg["intervention"]
    if a.rep_delay is not None:
        _was = int(iv["representative_arm"]["delay_days"])
        iv["representative_arm"]["delay_days"] = int(a.rep_delay)
        print(
            f"[setup] representative_arm.delay_days OVERRIDDEN {_was} -> {a.rep_delay} "
            f"(--rep-delay); greedy order and every dep_rep arm move with it",
            flush=True,
        )
        if not a.tag:
            print(
                "[setup] WARNING: --rep-delay without --tag overwrites the config-delay artefacts",
                flush=True,
            )
    n_draws = a.draws or iv["n_draws"]
    n_cont = a.conts if a.conts is not None else iv["n_continuations"]
    nw = a.workers or mp.cpu_count()
    streams = {s.strip() for s in a.streams.split(",")}
    OUT.mkdir(exist_ok=True)

    engine_ctx = E.build_context(cfg)
    shared_engine = engine_ctx["shared"]

    # ---- the PRIMARY ramp arm becomes the operator, before anything runs ---------------
    # `E.make_params` reads nat_scale_* out of `SHARED["fx"]`, which `build_context` fills from
    # config `fixed:`. Those config values are the in-window defaults, NOT the selected
    # forward ramp. Merging the primary arm into `fx` HERE means the default cannot reach
    # Params by any path: every stream, the pre-pass, the greedy order and the gates all
    # read the same merged dict, and workers get it through `shared_engine`.
    ramp_arms = iv["ramp_arms"]
    primary = next(x for x in ramp_arms if x["role"] == "primary")
    _before = {k: shared_engine["fx"].get(k) for k in RAMP_KEYS}
    shared_engine["fx"] = {**shared_engine["fx"], **{k: float(primary[k]) for k in RAMP_KEYS}}
    print(
        f"[setup] ramp arm PRIMARY={primary['name']}: "
        + "  ".join(f"{k}={shared_engine['fx'][k]}" for k in RAMP_KEYS)
        + f"   (config `fixed:` had {_before} -- OVERRIDDEN)",
        flush=True,
    )
    assert shared_engine["fx"]["nat_scale_t0"] == 88.0, (
        "nat_scale_t0 must stay 88 -- it is what makes phase 1 bit-identical across arms"
    )

    pm = cfg.get("pseudo_marginal", cfg.get("emcee", {}).get("pseudo_marginal", {}))
    shared_engine["pm_enabled"] = bool(pm.get("enabled", True))
    shared_engine["pm_M"] = int(pm.get("n_particles", 10))
    shared_engine["_max_inf"] = int(iv["max_infections"])
    E.init_worker(shared_engine)

    sfx = f"_{a.tag}" if a.tag else ""
    post_all = dict(np.load(HERE / (a.subsample or iv["subsample"])))
    split_seed = int(iv["seed"])
    post_greedy, post, npre, n_draws, greedy_rows, score_rows = split_greedy_and_score_draws(
        post_all,
        requested_score=int(n_draws),
        requested_pre=int(iv["greedy_n_pre"]),
        seed=split_seed,
    )
    if npre != int(iv["greedy_n_pre"]):
        print(
            f"[setup] WARNING smoke split shrank greedy_n_pre {int(iv['greedy_n_pre'])} -> {npre}",
            flush=True,
        )
    ctx = lib.build_ctx(cfg, shared_engine)
    print(
        f"[setup] Z={ctx['Z']}  served={int(ctx['served_mask'].sum())}  "
        f"cutoff={ctx['cutoff_day']}  horizon={ctx['horizon_day']} "
        f"(+{iv['horizon_days']}d)  draws={n_draws}  K={n_cont}  workers={nw}",
        flush=True,
    )
    print(f"[setup] streams: {sorted(streams)}", flush=True)

    # ---- pre-pass ensemble -> greedy placement order (built ONCE, shared) --
    t0 = time.time()
    pre_T, pre_P = [], []
    for j in range(npre):
        T, prm, _ = lib.build_tree(
            post_greedy, j, shared_engine, ctx["horizon_day"], cutoff_day=ctx["cutoff_day"], k=0
        )
        pre_T.append(T)
        pre_P.append(prm)
    print(
        f"[setup] pre-pass {len(pre_T)} trees in {time.time() - t0:.0f}s "
        f"(median n={int(np.median([t['n'] for t in pre_T])):,})",
        flush=True,
    )

    dep_rep = ctx["cutoff_day"] + int(iv["representative_arm"]["delay_days"])
    t0 = time.time()
    order = lib.greedy_order(
        pre_T, pre_P, ctx, iv, dep_rep, n_cand=int(iv["greedy_n_cand"]), kmax=max(iv["budgets"])
    )
    print(f"[setup] greedy order ({len(order)}) in {time.time() - t0:.0f}s", flush=True)
    print(f"[setup] greedy top-15: {[ctx['zones'][z] for z in order[:15]]}", flush=True)
    del pre_T, pre_P

    # ---- placement masks: greedy prefixes + fixed random draws ------------
    mach = {}
    for bud in iv["budgets"]:
        m = np.zeros(ctx["Z"], bool)
        m[order[:bud]] = True
        mach[("greedy", bud)] = m
        for rs in range(iv["n_random_seeds"]):
            r = np.zeros(ctx["Z"], bool)
            r[
                lib.random_placement(
                    bud, ~ctx["served_mask"], np.random.default_rng(9973 * (rs + 1) + bud)
                )
            ] = True
            mach[("random", bud, rs)] = r
    # Tagged like results/report: placement depends on the scenario variant and
    # split, so variants must not overwrite one another in place.
    (OUT / f"placement{sfx}.json").write_text(
        json.dumps(
            {
                "greedy_order": [int(z) for z in order],
                "greedy_zones": [ctx["zones"][z] for z in order],
                "greedy_zones_all": list(ctx["zones"]),  # full Z-length list, for MAPS labelling
                "eligible": [int(z) for z in np.where(~ctx["served_mask"])[0]],
                "random_masks": {
                    f"{b}_{rs}": [int(z) for z in np.where(mach[("random", b, rs)])[0]]
                    for b in iv["budgets"]
                    for rs in range(iv["n_random_seeds"])
                },
                "budgets": iv["budgets"],
            },
            indent=1,
        )
    )

    # ---- resume ------------------------------------------------------------
    path = OUT / f"results{sfx}.jsonl"
    report_path = OUT / f"report{sfx}.json"
    resume_meta = _resume_metadata(n_draws, n_cont, split_seed, score_rows)
    if a.fresh and path.exists():
        path.unlink()
    _check_resume_metadata(path, report_path, resume_meta)
    done = set()
    n_bad = 0
    if path.exists():
        with path.open() as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    if r.get("stream") == "base":
                        done.add((int(r["i"]), int(r.get("k", 0))))
                except (ValueError, KeyError, TypeError):
                    n_bad += 1
        print(f"[resume] {len(done)} (draw, continuation) tasks already complete", flush=True)
        if n_bad:
            print(
                f"[resume] {n_bad} unreadable lines skipped (likely a truncated final write)",
                flush=True,
            )

    tasks = [(i, k) for i in range(n_draws) for k in range(n_cont) if (i, k) not in done]
    print(f"[run] {len(tasks)} tasks", flush=True)

    shared = {
        "cfg": cfg,
        "iv": iv,
        "ctx": ctx,
        "post": post,
        "mach": mach,
        "streams": streams,
        "_engine": shared_engine,
    }
    t0 = time.time()
    ndone = 0
    write_metadata = not path.exists()
    with path.open("a") as fh:
        if write_metadata:
            fh.write(json.dumps(dict(stream="_metadata", **resume_meta)) + "\n")
        with mp.get_context("fork").Pool(nw, initializer=_init, initargs=(shared,)) as pool:
            for rows in pool.imap_unordered(_work, tasks, chunksize=1):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                ndone += 1
                if ndone % 50 == 0 or ndone == len(tasks):
                    el = time.time() - t0
                    print(
                        f"[run] {ndone}/{len(tasks)}  {el:.0f}s  "
                        f"eta {el / ndone * (len(tasks) - ndone):.0f}s",
                        flush=True,
                    )
    print(f"[done] {ndone} tasks in {time.time() - t0:.0f}s -> {path}", flush=True)
    (OUT / f"report{sfx}.json").write_text(
        json.dumps(
            {
                "n_draws": n_draws,
                "n_continuations": n_cont,
                "Z": ctx["Z"],
                "greedy_n_pre": npre,
                "posterior_rows_total": int(post_all["R0"].size),
                "split_seed": split_seed,
                "greedy_subsample_rows": [int(i) for i in greedy_rows],
                "score_subsample_rows": [int(i) for i in score_rows],
                "cutoff_day": ctx["cutoff_day"],
                "horizon_day": ctx["horizon_day"],
                "report_marks": iv["report_marks"],
                "streams": sorted(streams),
                "budgets": iv["budgets"],
                "modes": iv["modes"],
                "sdb_targets": iv["sdb_targets"],
                "sdb_scopes": iv["sdb_scopes"],
                "representative_arm": iv["representative_arm"],
                "greedy_top15": [ctx["zones"][z] for z in order[:15]],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
