"""One-off: TIME section re-based on the AVERTABLE denominator.

analysis.json's TIME reports av_total / n, where n is the draw's WHOLE epidemic
(history + forward). The rows already carry n_post_cut — burden still in the future at
the cutoff, i.e. what a decision-maker on the cutoff date can actually still avert —
so the avertable-denominator curve is recoverable from results.jsonl without re-running
the sweep.

Denominator is n_post_cut, NOT n_post_dep: n_post_cut is FIXED across delay arms, so the
curve isolates the cost of waiting. n_post_dep shrinks as the delay grows and would
partly cancel the very effect the figure is about.

Emits three series per (mode, budget, delay) cell:

  pct_inf     av_total  / n_post_cut  — % of avertable INFECTIONS  (the default figure)
  pct_deaths  avd_total / d_post_cut  — % of avertable DEATHS
  deaths      avd_total               — ABSOLUTE deaths averted

The absolute series carries the forward operator's saturation: absolute burden — and
therefore absolute deaths averted — is an UPPER BOUND. The two
percentage series do not, since the inflation is largely common to numerator and
denominator.

Streams the file (~800 MB), writes a small JSON. Stdlib only.
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


def arm_tag(r):
    """RDT/burial arm as a key fragment. `time` rows carry no scope and collapse to '',
    so the same code produces `local_b20` for the machines-only stream and
    `local_b20_all75` for the timerdt one — and the plot side can ask for either without
    knowing which stream it came from."""
    scope = r.get("scope")
    if scope in (None, "none") or r.get("sdb_target") is None:
        return "" if scope is None else "_none"
    tag = {"machine": "mach", "all": "all"}[scope]
    return f"_{tag}{round(100 * float(r['sdb_target']))}"


def main(path, dest, stream="time"):
    acc = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    want = (f'"stream": "{stream}"', f'"stream":"{stream}"')
    with open(path) as f:
        for ln in f:
            if want[0] not in ln and want[1] not in ln:
                continue
            r = json.loads(ln)
            ni, nd = r.get("n_post_cut", 0), r.get("d_post_cut", 0)
            if not ni:
                continue
            n_rows += 1
            cell = acc[(r["mode"], int(r["budget"]), arm_tag(r), int(r["delay"]))]
            cell["pct_inf"].append(100.0 * r["av_total"] / ni)
            cell["deaths"].append(float(r["avd_total"]))
            if nd:
                cell["pct_deaths"].append(100.0 * r["avd_total"] / nd)

    out = defaultdict(dict)
    for (mode, bud, arm, dl), cell in acc.items():
        out[f"{mode}_b{bud}{arm}"][str(dl)] = {
            **pctl(cell["pct_inf"]),
            "n": len(cell["pct_inf"]),
            **{f"{s}_{k}": v for s in ("pct_deaths", "deaths") for k, v in pctl(cell[s]).items()},
        }
    for k in list(out):
        per = out[k]
        base = per.get("0", {}).get("p50")
        if base:
            out[k + "_rel_to_day0"] = {
                d: round(v["p50"] / base, 4) for d, v in per.items() if v["p50"]
            }

    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[time_avertable] {n_rows} {stream} rows -> {dest}")
    keys = sorted(k for k in out if "_rel" not in k and k.startswith("local_b"))
    for lab, key in [
        ("% avertable infections", "p50"),
        ("% avertable deaths", "pct_deaths_p50"),
        ("absolute deaths averted", "deaths_p50"),
    ]:
        print(f"\n  {lab} (LOCAL)")
        for k in keys:
            per = out[k]
            print(
                f"    {k[6:]:<12}", " ".join(f"{per[d][key]:8.2f}" for d in sorted(per, key=int))
            )
        print(f"    {'days':<12}", " ".join(f"{int(d):8d}" for d in sorted(per, key=int)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
