"""Stage 0 — extract a thinned subsample from the calibration posterior.

The source chain is a large RAW (post-burn but UNTHINNED) posterior. Every later stage
reads the small npz this script writes, not the full chain.

Sampling design:
  - emcee flattens as index = step * n_walkers + walker, so a stride co-prime to
    n_walkers (480) spreads draws across BOTH walkers and chain time. A stride that
    is a multiple of 480 would lock onto a single walker.
  - Full post-burn chain, not the tail half: this maximises ESS. The cost is a little
    residual drift in R0 across chain deciles.
  - Also carries pm_logL for the SELECTED seed, so a replay can be checked against it.

Run:  uv run python extract_subsample.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npf
import yaml

HERE = Path(__file__).parent
N_WALKERS = 480  # must match the source chain's emcee.n_walkers
BURN = -50_000.0  # burnout logL threshold from the shared engine


def _npy_header(fh):
    v = npf.read_magic(fh)
    return npf.read_array_header_1_0(fh) if v == (1, 0) else npf.read_array_header_2_0(fh)


def stream_take(zf: zipfile.ZipFile, name: str, want: np.ndarray):
    """Pull specific row indices out of a (deflate-compressed) .npy member of an npz
    without materialising the whole column. `want` must be sorted ascending."""
    with zf.open(name) as fh:
        shape, fortran, dtype = _npy_header(fh)
        assert not fortran, name
        ncols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        rowbytes = ncols * dtype.itemsize
        nrows = shape[0]
        out = np.empty((want.size, ncols), dtype=dtype)
        buf = b""
        row = 0
        wi = 0
        BLOCK = 16384
        while row < nrows and wi < want.size:
            n = min(BLOCK, nrows - row)
            need = n * rowbytes
            while len(buf) < need:
                chunk = fh.read(1 << 22)
                if not chunk:
                    break
                buf += chunk
            blk = np.frombuffer(buf[:need], dtype=dtype).reshape(n, ncols)
            buf = buf[need:]
            hi = row + n
            while wi < want.size and want[wi] < hi:
                out[wi] = blk[want[wi] - row]
                wi += 1
            row = hi
        assert wi == want.size, f"{name}: only filled {wi}/{want.size}"
    return out.reshape(-1) if len(shape) == 1 else out, nrows


_SEED_OVERRIDE = None
_OUT_OVERRIDE = None


def main(n_target: int):
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    iv = cfg["intervention"]
    src = (HERE / iv["posterior"]).resolve()
    out_path = HERE / (_OUT_OVERRIDE or iv["subsample"])
    out_path.parent.mkdir(exist_ok=True)

    zf = zipfile.ZipFile(src)
    members = {n[:-4] for n in zf.namelist()}
    print(f"[stage0] source {src}")
    print(f"[stage0] members: {sorted(members)}")

    # ---- total length from the R0 header (cheap: header only)
    with zf.open("R0.npy") as fh:
        shape, _, _ = _npy_header(fh)
    N = int(shape[0])

    # ---- two source kinds, and the difference is NOT cosmetic --------------------------
    # `raw`    the untouched emcee chain, N = 480 walkers x n_steps in walker-major order.
    #          Thin by a stride CO-PRIME to N_WALKERS so the draws spread over walkers AND
    #          time; a stride sharing a factor with 480 would sample a walker subset.
    # `pooled` a source that has ALREADY been thinned upstream (e.g. a 25,000-draw uniform
    #          subsample drawn at its own seed). The walker-major
    #          structure is GONE, so the co-prime stride is meaningless on it -- and worse,
    #          it degenerates SILENTLY: at N=25,000 and n=4,000 the stride search walks
    #          6->5->4->3->2->1 and lands on 1, i.e. "take the first 4,000 rows", while
    #          still printing a confident walker-spread diagnostic computed on a structure
    #          that does not exist. Uniform choice at the configured seed instead.
    # ---- when the source is a MID-RUN CHECKPOINT, three things change ------------------
    # `checkpoint_ppc.npz` is written by the calibration run each chunk via
    # emcee's get_chain(discard=, thin=, flat=True). That call keeps EVERY walker and strides
    # the STEP axis, so the flattened row index is step*n_walkers + walker -- the walker-major
    # structure is intact and `kind: raw` is the correct reader (see config.yaml).
    #
    # What the checkpoint does NOT carry, and what this script does about it:
    #   pm_seeds / pm_logL   the per-theta particle seeds and their log-likelihoods. Absent,
    #                        so `sel_col`/`sel_logL` cannot be recovered and the selection
    #                        diagnostic cannot be computed. Both are SKIPPED, and the meta
    #                        records `g3: null` with a reason rather than silently omitting it.
    #   zones                the 470-zone name vector. Taken from the shared engine context
    #                        instead, which is where the engine itself gets it, so the zone
    #                        name list can still be checked.
    #   f_refer              FIXED at cfg.fixed, never sampled. A finished posterior.npz
    #                        stores it as a constant column purely for downstream
    #                        compatibility; reconstructed the same way here, from the
    #                        same config key.
    _is_ckpt = "pm_logL" not in members
    kind = iv.get("posterior_pool", "raw")
    if _is_ckpt:
        print(
            "[stage0] CHECKPOINT source: no pm_logL member -> the replay-vs-stored and "
            "selection checks are UNAVAILABLE by construction, not by omission."
        )
        if "_iter" in members:
            it = int(np.load(src)["_iter"][0])
            tau = float(np.load(src)["_tau_max"][0])
            print(
                f"[stage0] checkpoint taken at iter {it:,}  tau_max {tau:,.1f}  "
                f"({it / 60000:.1%} of the 60,000-step target) -- NOT a converged posterior"
            )
    if kind == "raw":
        n_steps = N // N_WALKERS
        print(f"[stage0] source kind=raw  N={N:,} = {N_WALKERS} walkers x {n_steps:,} steps")
        stride = N // n_target
        while np.gcd(stride, N_WALKERS) != 1:
            stride -= 1
        idx = (np.arange(n_target, dtype=np.int64) * stride) % N
        idx = np.unique(idx)
        print(
            f"[stage0] stride={stride:,} (gcd with {N_WALKERS} = "
            f"{np.gcd(stride, N_WALKERS)}); n_draws={idx.size}"
        )
        w_touched = np.unique(idx % N_WALKERS).size
        s_span = (idx // N_WALKERS).max() - (idx // N_WALKERS).min()
        print(
            f"[stage0] spread: {w_touched}/{N_WALKERS} walkers touched, "
            f"step span {s_span:,}/{n_steps:,}"
        )
    elif kind == "pooled":
        seed = int(_SEED_OVERRIDE if _SEED_OVERRIDE is not None else iv["seed"])
        idx = np.sort(np.random.default_rng(seed).choice(N, size=min(n_target, N), replace=False))
        print(f"[stage0] source kind=pooled  N={N:,} (already thinned upstream)")
        print(
            f"[stage0] uniform without replacement at intervention.seed={seed}; n_draws={idx.size}"
        )
        # The replicate differs ONLY in this seed, so the pair measures subsample noise as
        # well as forward noise.
    else:
        raise ValueError(f"intervention.posterior_pool must be 'raw' or 'pooled'; got {kind!r}")

    # CAL comes from the shared engine, not a hardcoded list: the calibrated parameter set
    # changes as parameters open and close, and a stale list would silently drop or invent
    # parameters. Deriving it means the extractor tracks whatever posterior it is pointed at.
    sys.path.insert(0, str(HERE.parent))
    from bdbv import engine as E

    CAL = list(E.CAL)
    print(f"[stage0] CAL from shared engine ({len(CAL)}): {CAL}")
    assert set(CAL).issubset(members), sorted(set(CAL) - members)

    post = {}
    for k in CAL + ["seed_day", "sim_seed"]:
        col, _ = stream_take(zf, f"{k}.npy", idx)
        post[k] = col
        print(f"[stage0]   {k:24s} ok")

    if _is_ckpt:
        # f_refer: FIXED, reconstructed from the same config key the engine writes it
        # from. Not read by any stream; carried for schema compatibility only.
        post["f_refer"] = np.full(idx.size, float(cfg["fixed"].get("f_refer", 0.05)))
        g3 = None
        print(
            "[stage0]   f_refer                  synthesised from cfg.fixed "
            f"({post['f_refer'][0]:.3f}) -- FIXED param, not sampled"
        )
    else:
        col, _ = stream_take(zf, "f_refer.npy", idx)
        post["f_refer"] = col
        pm_seeds, _ = stream_take(zf, "pm_seeds.npy", idx)
        pm_logL, _ = stream_take(zf, "pm_logL.npy", idx)
        post["pm_seeds"] = pm_seeds
        post["pm_logL"] = pm_logL

        # ---- recover which of the M seeds sim_seed IS, and its stored logL.
        # This is the target a replay must reproduce.
        match = pm_seeds == post["sim_seed"][:, None]
        nmatch = match.sum(axis=1)
        sel = np.argmax(match, axis=1)
        post["sel_col"] = sel.astype(np.int64)
        post["sel_logL"] = pm_logL[np.arange(sel.size), sel]

        # ---- re-check within-theta selection on the extracted subsample
        mx = pm_logL.max(axis=1, keepdims=True)
        w = np.where(np.isfinite(pm_logL), np.exp(pm_logL - mx), 0.0)
        w /= w.sum(axis=1, keepdims=True)
        w_sel = w[np.arange(sel.size), sel]
        ess_u = 1.0 / (w**2).sum(axis=1)
        g3 = {
            "n": int(sel.size),
            "sim_seed_matched_frac": float((nmatch == 1).mean()),
            "raw_burnout_frac": float((pm_logL < BURN).mean()),
            "selected_burnout_frac": float((post["sel_logL"] < BURN).mean()),
            "all_M_burnout_frac": float((pm_logL.max(axis=1) < BURN).mean()),
            "mean_surviving_seeds": float((pm_logL >= BURN).sum(axis=1).mean()),
            "seed_ess_mean": float(ess_u.mean()),
            "seed_ess_p95": float(np.percentile(ess_u, 95)),
            "selected_is_argmax_frac": float((sel == np.argmax(pm_logL, axis=1)).mean()),
            "corr_thetaLL_vs_wsel": float(
                np.corrcoef(np.log(np.exp(pm_logL - mx).mean(axis=1)) + mx.ravel(), w_sel)[0, 1]
            ),
        }
        print("\n[stage0] within-theta selection on the subsample:")
        for k, v in g3.items():
            print(f"           {k:26s} {v}")

    if "zones" in members:
        zones, _ = stream_take(zf, "zones.npy", np.arange(470, dtype=np.int64))
    else:
        # The checkpoint stores theta only. Take the zone vector from the engine's own
        # shared context -- the same object the engine hands the simulator -- so the zone
        # name check still runs against the list the run will actually use.
        zones = np.asarray(E.build_context(cfg)["shared"]["zones"], dtype=str)
        print(f"[stage0]   zones                    from engine context ({zones.size})")
    post["zones"] = np.asarray(zones, dtype=str)
    post["src_index"] = idx

    np.savez_compressed(out_path, **post)
    meta = dict(
        source=str(src),
        source_kind=kind,
        source_N=N,
        n_draws=int(idx.size),
        # walker-structure diagnostics exist only for a raw chain; a pooled source
        # has none, and reporting them would describe a structure that isn't there.
        **(
            {
                "n_walkers": N_WALKERS,
                "n_steps": n_steps,
                "stride": int(stride),
                "walkers_touched": int(w_touched),
                "step_span": int(s_span),
            }
            if kind == "raw"
            else {"select_seed": int(seed)}
        ),
        CAL=CAL,
        g3=g3,
        # Recorded so downstream reporting cannot quote these numbers as if they
        # came off a finished chain. `checkpoint` is True whenever pm_logL was absent.
        checkpoint=bool(_is_ckpt),
        checkpoint_iter=(int(np.load(src)["_iter"][0]) if _is_ckpt else None),
        checkpoint_tau_max=(float(np.load(src)["_tau_max"][0]) if _is_ckpt else None),
        g3_unavailable_reason=(
            None
            if g3 is not None
            else "source is a mid-run checkpoint: pm_seeds/pm_logL are "
            "not written to checkpoint_ppc.npz, so within-theta "
            "selection cannot be re-reported"
        ),
        medians={k: float(np.median(post[k])) for k in CAL},
    )
    # The meta follows the OUTPUT name, not a fixed one: with --out but a fixed meta path,
    # building replicate B silently overwrites replicate A's provenance record.
    meta_path = out_path.with_name(out_path.stem.replace("sq_draws", "subsample_meta") + ".json")
    meta_path.write_text(json.dumps(meta, indent=1))
    sz = out_path.stat().st_size / 1e6
    print(f"\n[stage0] wrote {out_path.name} ({sz:.1f} MB) + {meta_path.name}")
    # Compare against the source chain's own full-chain medians, read from its report.json
    # rather than pasted in -- the check is that the thinned subsample is representative of
    # the chain it came from, so a hardcoded reference would defeat it after any refit.
    # A mid-run checkpoint has no report.json (that is written at finalize), so the
    # cross-check is skipped rather than silently compared against something else.
    src_report = src.parent / "report.json"
    if src_report.exists():
        ref = json.loads(src_report.read_text()).get("medians", {})
        print(f"[stage0] medians vs {src_report.parent.parent.name} full-chain report.json:")
        worst = 0.0
        for k in CAL:
            m = float(np.median(post[k]))
            if ref.get(k):
                r = float(ref[k])
                rat = m / r
                worst = max(worst, abs(rat - 1.0))
                flag = "   <-- CHECK" if abs(rat - 1.0) > 0.05 else ""
                print(f"           {k:24s} {m:>11.4f}  vs {r:>11.4f}   ({rat:.3f}x){flag}")
            else:
                print(f"           {k:24s} {m:>11.4f}  (no reference)")
        print(
            f"[stage0] worst median deviation: {100 * worst:.1f}%"
            f" ({'OK' if worst < 0.05 else 'INVESTIGATE'})"
        )
    else:
        print(f"[stage0] no report.json at {src_report} -- skipping median cross-check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "number of posterior rows to extract; production runs need "
            "intervention.n_draws + intervention.greedy_n_pre so placement can be "
            "trained and evaluated on disjoint draws"
        ),
    )
    # Replicate: a different selection seed draws a DIFFERENT subsample from the same
    # pool, so the replicate pair captures subsample noise as well as forward noise.
    ap.add_argument("--seed", type=int, default=None, help="override intervention.seed")
    ap.add_argument("--out", type=str, default=None, help="override intervention.subsample")
    a = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    _SEED_OVERRIDE, _OUT_OVERRIDE = a.seed, a.out
    iv = cfg["intervention"]
    main(a.n or (int(iv["n_draws"]) + int(iv["greedy_n_pre"])))
