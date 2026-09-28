"""One-off: PLACE section re-based on the AVERTABLE denominator.

Companion to time_avertable.py, same argument and the same denominator. analysis.json's
PLACE reports av_total / n, where n is the draw's WHOLE epidemic (history + forward);
a decision-maker standing on the cutoff date can only avert what is still ahead, so the
honest denominator is n_post_cut. The rows already carry it, so this is recoverable from
results.jsonl without re-running the sweep.

Quantities mirror analyze.q_place exactly — greedy, random, and the paired gap (greedy
minus that draw's OWN mean random) — with n_post_cut swapped in for n everywhere. Cross-
check: local greedy at delay 0 here must equal time_avertable.json's local_b*['0'],
because run.py builds both from the same greedy mask at the same deployment day.

Streams the file (~400 MB for the 30-day sweep), writes a small JSON. Stdlib only.

Run:  uv run python place_avertable.py outputs/results_d30.jsonl \
          outputs/place_avertable_d30_pct_deaths.json pct_deaths
"""

import json
import math
import sys
from collections import defaultdict

Q = (5, 25, 50, 75, 95)


def pctl(x):
    """numpy.percentile's default 'linear' interpolation, in stdlib."""
    x = sorted(v for v in x if math.isfinite(v))
    if not x:
        return {f"p{q}": None for q in Q}
    out = {}
    for q in Q:
        h = (len(x) - 1) * q / 100.0
        lo = int(h)
        hi = min(lo + 1, len(x) - 1)
        out[f"p{q}"] = round(x[lo] + (h - lo) * (x[hi] - x[lo]), 4)
    return out


def main(path, dest, metric="pct_inf"):
    """`metric` selects the numerator/denominator PAIR.

        pct_inf     av_total  / n_post_cut   — % of avertable INFECTIONS (the original)
        pct_deaths  avd_total / d_post_cut   — % of avertable DEATHS

    Both series are offered so the place and time figures cannot disagree about what
    their identically-worded y-axis means. The numerator and denominator are taken as a
    PAIR -- mixing avd_total with n_post_cut would silently produce a deaths-per-infection
    ratio scaled by 100.
    """
    if metric not in ("pct_inf", "pct_deaths"):
        raise SystemExit(f"metric must be pct_inf or pct_deaths, got {metric!r}")
    num_key, den_key = (
        ("av_total", "n_post_cut") if metric == "pct_inf" else ("avd_total", "d_post_cut")
    )

    # per (mode, budget): greedy pct by draw, random pct by draw (several seeds each)
    g_by_key = defaultdict(dict)  # key -> {i: pct}
    r_by_key = defaultdict(lambda: defaultdict(list))  # key -> {i: [pct, ...]}
    n_rows = 0
    n_skipped_zero_den = 0
    with open(path) as f:
        for ln in f:
            if '"stream": "place"' not in ln and '"stream":"place"' not in ln:
                continue
            r = json.loads(ln)
            d = r.get(den_key, 0)
            if not d:
                # On the deaths metric this is a REAL population, not a parse failure: a
                # draw can have forward infections but zero forward deaths. Counted so the
                # n_draws difference between the two metrics is visible rather than silent.
                n_skipped_zero_den += 1
                continue
            n_rows += 1
            key = f"{r['mode']}_b{int(r['budget'])}"
            pct = 100.0 * r[num_key] / d
            if r["strategy"] == "greedy":
                g_by_key[key][r["i"]] = pct
            else:
                r_by_key[key][r["i"]].append(pct)

    out = {}
    for key in sorted(g_by_key, key=lambda k: (k.split("_b")[0], int(k.split("_b")[1]))):
        g, rnd = g_by_key[key], r_by_key[key]
        if not rnd:
            continue
        rmean = {i: sum(v) / len(v) for i, v in rnd.items()}
        gap = [g[i] - rmean[i] for i in g if i in rmean]
        flat = [v for vs in rnd.values() for v in vs]
        out[key] = {
            "greedy": pctl(list(g.values())),
            "random": pctl(flat),
            "paired_gap_pp": pctl(gap),
            "greedy_beats_random_frac": (
                round(sum(v > 0 for v in gap) / len(gap), 4) if gap else None
            ),
            "n_draws": len(g),
        }

    out["_meta"] = {
        "metric": metric,
        "numerator": num_key,
        "denominator": den_key,
        "rows_used": n_rows,
        "rows_skipped_zero_denominator": n_skipped_zero_den,
    }
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(
        f"[place_avertable] metric={metric} ({num_key}/{den_key})  {n_rows} place rows -> {dest}"
    )
    if n_skipped_zero_den:
        print(f"[place_avertable] {n_skipped_zero_den} rows skipped for {den_key} == 0")
    for k, v in out.items():
        if k.startswith("local_"):
            print(
                f"  {k:<12} greedy {v['greedy']['p50']:6.2f}  "
                f"random {v['random']['p50']:6.2f}  gap {v['paired_gap_pp']['p50']:6.2f}"
            )


if __name__ == "__main__":
    main(*sys.argv[1:4])
