"""Absolute deaths averted, against the still-avertable denominator.

Streams results.jsonl one line at a time and keeps only scalars, so it runs in seconds
and constant memory. `analyze.py` materialises every row (including 20,000 per-zone MAPS
arrays, ~28M numbers), which needs far more memory than a laptop has.

WHY a different denominator: under the status quo roughly three quarters of the epidemic
is already in the past at the cutoff. A fraction of TOTAL burden therefore understates the
intervention ~4x for a reason that has nothing to do with how good the intervention is.
Two honest denominators:
    post-cutoff  what is still ahead when the DECISION is made
    post-deploy  what is still ahead when the machines actually LAND
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KEEP = {"base", "place", "pm", "time", "casc"}

# Replicate tag: `--tag b` reads results_b.jsonl and writes avertable_b.json,
# matching run.py's suffix.
SFX = ""


def pctl(v, q=(5, 25, 50, 75, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{i}": float(np.percentile(v, i)) for i in q} if v.size else {}


def main():
    base, arms = {}, defaultdict(list)
    n_bad = 0
    with (HERE / "outputs" / f"results{SFX}.jsonl").open() as f:
        for ln in f:
            if '"maps"' in ln:  # skip the heavy per-zone rows
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            st = r.get("stream")
            if st not in KEEP or r.get("n", 0) == 0:
                continue
            if st == "base":
                if r.get("k", 0) == 0 and "d_post_cut" in r:
                    base[r["i"]] = r
                continue
            key = None
            if st == "place" and r["strategy"] == "greedy" and r["mode"] == "local":
                key = f"machines greedy local b{r['budget']}"
            elif st == "pm" and r["scope"] == "all" and r.get("sdb_target"):
                key = f"machines + burial {int(100 * r['sdb_target'])}%"
            elif st == "time" and r["mode"] == "local" and r["budget"] == 35:
                key = f"delay {r['delay']:>3d}d (machines, b35)"
            elif st == "casc" and r["channel"] == "full":
                key = "full package (rep arm)"
            if key:
                arms[key].append(r)

    if n_bad:
        print(f"[warn] {n_bad} unparseable lines (likely a truncated final write)")
    print(f"[avertable] {len(base)} baseline draws, {len(arms)} arms\n")
    bd = [r for r in base.values()]
    tot_d = pctl([r["deaths"] for r in bd])
    past_d = pctl([r["d_at_cutoff"] for r in bd])
    fut_d = pctl([r["d_post_cut"] for r in bd])
    frac = pctl([100 * r["n_at_cutoff"] / r["n"] for r in bd])
    print("=" * 88)
    print("THE DENOMINATOR — most of this epidemic precedes the decision")
    print("=" * 88)
    print(
        f"  total deaths, whole epidemic      {tot_d['p50']:>7,.0f}  [{tot_d['p5']:,.0f}, {tot_d['p95']:,.0f}]"
    )
    print(
        f"  already locked in at the cutoff   {past_d['p50']:>7,.0f}  [{past_d['p5']:,.0f}, {past_d['p95']:,.0f}]"
    )
    print(
        f"  STILL AVERTABLE                   {fut_d['p50']:>7,.0f}  [{fut_d['p5']:,.0f}, {fut_d['p95']:,.0f}]"
    )
    print(f"  epidemic already past             {frac['p50']:>6.0f}%")
    print()
    print("=" * 88)
    print("DEATHS AVERTED — absolute, and against what was still preventable")
    print("=" * 88)
    print(f"  {'arm':34s} {'deaths averted':>26s} {'%avertable':>11s} {'%total':>8s}")
    order = sorted(arms, key=lambda k: (k.startswith("delay"), k))
    rows_out = {}
    for k in order:
        sel = arms[k]
        dav = [r["avd_total"] for r in sel]
        D = pctl(dav)
        pc = pctl(
            [100 * r["avd_total"] / r["d_post_cut"] for r in sel if r.get("d_post_cut", 0) > 0]
        )
        pd_ = pctl(
            [100 * r["avd_total"] / r["d_post_dep"] for r in sel if r.get("d_post_dep", 0) > 0]
        )
        pt = pctl(
            [
                100 * r["avd_total"] / base[r["i"]]["deaths"]
                for r in sel
                if r["i"] in base and base[r["i"]]["deaths"] > 0
            ]
        )
        rows_out[k] = {
            "deaths_averted": D,
            "pct_avertable_postcut": pc,
            "pct_avertable_postdeploy": pd_,
            "pct_total": pt,
            "n": len(sel),
        }
        print(
            f"  {k:34s} {D['p50']:>6,.0f} [{D['p5']:>4,.0f}, {D['p95']:>6,.0f}]"
            f" {pc.get('p50', float('nan')):>10.1f}% {pt.get('p50', float('nan')):>7.2f}%"
        )

    # Brake stratification on absolute deaths averted, on `R_floor` and NOT `nat_half`:
    # the latter is FIXED at 0.0 here, which would make every quartile edge 0.0 and land
    # every draw in Q1, silently. The stratified table IS the result here, not the pooled
    # figure -- averted burden depends strongly on where in the brake posterior a draw sits.
    BKEY = "R_floor"
    nh = np.array([base[i][BKEY] for i in base])
    qs = np.percentile(nh, [25, 50, 75])
    print()
    print("=" * 88)
    print(f"BY FORWARD TRANSMISSION FLOOR ({BKEY} quartiles: {qs.round(3).tolist()})")
    print("=" * 88)
    tgt = "machines + burial 75%"
    if tgt in arms:
        print(f"  arm: {tgt}")
        print(f"  {'brake':26s} {'deaths averted':>26s} {'%avertable':>11s}")
        bins = [
            ("Q1 strongest", lambda v: v <= qs[0]),
            ("Q2", lambda v: qs[0] < v <= qs[1]),
            ("Q3", lambda v: qs[1] < v <= qs[2]),
            ("Q4 weakest", lambda v: v > qs[2]),
        ]
        for bn, bf in bins:
            sel = [r for r in arms[tgt] if r["i"] in base and bf(base[r["i"]][BKEY])]
            if not sel:
                continue
            D = pctl([r["avd_total"] for r in sel])
            pc = pctl(
                [100 * r["avd_total"] / r["d_post_cut"] for r in sel if r.get("d_post_cut", 0) > 0]
            )
            print(
                f"  {bn:26s} {D['p50']:>6,.0f} [{D['p5']:>4,.0f}, {D['p95']:>6,.0f}]"
                f" {pc.get('p50', float('nan')):>10.1f}%"
            )
            rows_out[f"{tgt} | {bn}"] = {
                "deaths_averted": D,
                "pct_avertable_postcut": pc,
                "n": len(sel),
            }

    out = {
        "denominator": {
            "total_deaths": tot_d,
            "deaths_already_at_cutoff": past_d,
            "deaths_still_avertable": fut_d,
            "pct_epidemic_past": frac,
        },
        "arms": rows_out,
        "brake_key": BKEY,
        "brake_quartiles": [float(q) for q in qs],
    }
    (HERE / "outputs" / f"avertable{SFX}.json").write_text(json.dumps(out, indent=1))
    print(f"\n[avertable] -> outputs/avertable{SFX}.json")


if __name__ == "__main__":
    import argparse as _argparse

    _ap = _argparse.ArgumentParser()
    _ap.add_argument("--tag", type=str, default="", help="artefact suffix, matches run.py --tag")
    _a = _ap.parse_args()
    SFX = f"_{_a.tag}" if _a.tag else ""
    main()
