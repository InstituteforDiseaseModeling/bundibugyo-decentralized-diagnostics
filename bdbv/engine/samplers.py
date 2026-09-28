"""Calibration drivers: emcee ensemble MCMC (with HDF5 backend backup and repair) and
prior-predictive SIR, each followed by posterior-predictive checks and figures."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from ..model.priors import log_prior, sample_prior
from .context import build_context
from .figures import (
    fig_all_in_one,
    fig_capacity_iso,
    fig_likelihood_diagnosis,
    fig_posterior,
    fig_ppc,
    fig_scored_only,
    fig_spatial,
)
from .scoring import ess_of, log_likelihood, ppc_draw, score_draw, select_within_theta
from .state import (
    CAL,
    FIXED_FROM_CAL,
    HERE,
    OUT,
    SHARED,
    ensure_dirs,
    init_worker,
    pm_rng,
    prior_support,
)

# ---------------------------------------------------------------------------
# emcee ensemble MCMC, PSEUDO-MARGINAL. Freezing sim_seed to one constant would make
# L(theta) deterministic, but that conditions inference on a single latent realisation
# and creates flat "burnout" plateaus that trap walkers. Instead M genuinely random
# seeds are drawn per evaluation and scored as L_hat = (1/M) sum_m p(data|theta,u_m),
# an unbiased estimator, so emcee targets the EXACT posterior (Andrieu & Roberts 2009).
# emcee caches each walker's current log-prob/blobs and only recomputes on the
# proposal, which realises the pseudo-marginal persist-u-on-accept rule for
# free; the M seeds + their per-particle logL are returned as a blob.
# ---------------------------------------------------------------------------


def _emcee_log_prob(theta) -> tuple[float, np.ndarray]:
    """Pseudo-marginal log-posterior for emcee. `theta` is a flat array in CAL
    order. Returns (log_posterior, blob) where blob is a float64 array of length
    2M: [seed_0..seed_{M-1}, logL_0..logL_{M-1}]. seed_day is derived per-theta
    from calibrated `seed_day_offset`; f_refer is read from cfg.fixed.

    With pm disabled, M=1 and the seed is the frozen `_emcee_sim_seed`, still returned
    as a blob for a consistent chain layout.
    """
    M = int(SHARED.get("pm_M", 1))
    lp = 0.0
    p = {k: float(SHARED["fx"][k]) for k in FIXED_FROM_CAL}  # see score_draw
    for k, v in zip(CAL, theta):
        lp_k = log_prior(SHARED["priors"][k], float(v))
        if not np.isfinite(lp_k):
            # consistent blob shape on prior-reject; logL = -inf so it can never
            # be mistaken for a good realisation if ever inspected.
            return -np.inf, np.concatenate([np.zeros(M), np.full(M, -np.inf)])
        lp += lp_k
        p[k] = float(v)
    seed_day = round(SHARED["start_mean"] + p["seed_day_offset"])
    if SHARED.get("pm_enabled", False):
        # fresh, theta-independent seeds (int fits exactly in float64 for storage)
        seeds = pm_rng().integers(0, 2**53, size=M).astype(np.int64)
    else:
        seeds = np.full(M, int(SHARED["_emcee_sim_seed"]), dtype=np.int64)
    logL = np.full(M, -np.inf, dtype=np.float64)
    for m in range(M):
        try:
            val = log_likelihood(p, seed_day, int(seeds[m]))
            logL[m] = val if np.isfinite(val) else -np.inf
        except (ValueError, ArithmeticError):  # numerical failure = zero likelihood;
            logL[m] = -np.inf  # anything else is a bug and should surface
    blob = np.concatenate([seeds.astype(np.float64), logL])
    finite = np.isfinite(logL)
    if not finite.any():
        return -np.inf, blob
    # log_mean_exp: log( (1/M) sum_m exp(logL_m) ); burnout particles contribute 0.
    mx = logL[finite].max()
    logLhat = mx + np.log(np.exp(logL[finite] - mx).sum()) - np.log(M)
    total = lp + logLhat
    if not np.isfinite(total):
        return -np.inf, blob
    return total, blob


# ---------------------------------------------------------------------------
# spot-reclaim-hardened HDF5 backend.
# A spot reclamation that kills emcee mid-checkpoint-write leaves emcee_chain.h5
# physically shorter than the EOF address in its HDF5
# superblock — h5py then refused to open it, so the watchdog's relaunch would
# crash-loop instead of resuming. These helpers make every (re)launch self-heal
# BEFORE the backend is touched:
#   1. rotating backups snapshot a consistent checkpoint at each chunk boundary,
#   2. a truncated live file is zero-padded up to its superblock EOF (automating
#      what would otherwise be a manual recovery),
#   3. if that fails, the newest readable backup is restored.
# The watchdog itself is unchanged — it just relaunches run.py.
# ---------------------------------------------------------------------------


def _rotate_backup(chain_h5, keep: int = 3):
    """Snapshot a consistent checkpoint at a chunk boundary, keeping `keep`
    rotations (.bak0 = newest). emcee has just flushed, so the file is valid."""
    import shutil

    try:
        for i in range(keep - 1, 0, -1):
            src = Path(str(chain_h5) + f".bak{i - 1}")
            if src.exists():
                shutil.copy2(src, Path(str(chain_h5) + f".bak{i}"))
        shutil.copy2(chain_h5, Path(str(chain_h5) + ".bak0"))
    except OSError as ex:  # backups are best-effort
        print(f"[h5-repair] backup warning: {ex}", flush=True)


def _pad_h5_to_eof(path) -> bool:
    """Zero-pad a physically-truncated HDF5 file up to the EOF address declared
    in its superblock. Returns True if it padded. Superblock EOF-address offset:
    v0 -> 24+2*sizeof_offsets, v1 -> 28+2*sizeof_offsets, v2/3 -> 12+2*sizeof."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head[:8] != b"\x89HDF\r\n\x1a\n":
            return False
        ver = head[8]
        if ver in (0, 1):
            off = head[13]
            eof_off = (24 if ver == 0 else 28) + 2 * off
        elif ver in (2, 3):
            off = head[9]
            eof_off = 12 + 2 * off
        else:
            return False
        if len(head) < eof_off + off:
            return False
        eof = int.from_bytes(head[eof_off : eof_off + off], "little")
        cur = path.stat().st_size
        if eof > cur:
            with open(path, "ab") as f:
                f.write(b"\x00" * (eof - cur))
            return True
        return False
    except (OSError, IndexError):
        return False


def _backend_iteration(backend) -> int:
    """Steps stored in an emcee backend. `HDFBackend.iteration` is read from an h5py
    attribute, which h5py types as `Empty | ndarray`; the stored value is a scalar int."""
    return int(backend.iteration)


def _open_or_repair_backend(chain_h5, name: str = "mcmc"):
    """Return an emcee HDFBackend that opens cleanly, repairing a truncated file
    or restoring the newest good backup first. Runs at every (re)launch."""
    import glob
    import os
    import shutil

    import emcee

    chain_h5 = Path(chain_h5)

    def _try(path):
        try:
            b = emcee.backends.HDFBackend(str(path), name=name)
            _ = b.iteration  # forces a read of the file
            if _backend_iteration(b) > 0:
                _ = b.shape
            return b
        except (OSError, KeyError, RuntimeError, ValueError):
            return None

    if not chain_h5.exists():
        return emcee.backends.HDFBackend(str(chain_h5), name=name)  # fresh start
    b = _try(chain_h5)
    if b is not None:
        return b
    print(f"[h5-repair] {chain_h5.name} unreadable — attempting recovery", flush=True)
    if _pad_h5_to_eof(chain_h5):
        b = _try(chain_h5)
        if b is not None:
            print(
                f"[h5-repair] zero-padded to superblock EOF; resume at iter {b.iteration}",
                flush=True,
            )
            return b
    for bak in sorted(glob.glob(str(chain_h5) + ".bak*"), key=os.path.getmtime, reverse=True):
        if _try(Path(bak)) is not None:
            shutil.copy2(bak, chain_h5)
            b = _try(chain_h5)
            if b is not None:
                print(
                    f"[h5-repair] restored from {Path(bak).name}; resume at iter {b.iteration}",
                    flush=True,
                )
                return b
    print("[h5-repair] WARNING: unrepairable and no good backup — starting FRESH", flush=True)
    try:
        chain_h5.unlink()
    except FileNotFoundError:
        pass
    return emcee.backends.HDFBackend(str(chain_h5), name=name)


def main_emcee(
    steps_override=None,
    walkers_override=None,
    workers_override=None,
    init_npz: str | None = None,
    burn_override: int | None = None,
) -> None:
    """Entry point for `--sampler emcee`. Runs emcee ensemble MCMC and saves
    a posterior.npz compatible with the SIR figure code.

    Writes outputs/posterior.npz.
    """
    ensure_dirs()
    from multiprocessing import Pool

    import emcee

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    ctx = build_context(cfg)
    shared = ctx["shared"]
    em = cfg.get("emcee", {})
    n_walkers = walkers_override or int(em.get("n_walkers", 240))
    n_steps = steps_override or int(em.get("n_steps", 5000))
    n_burn = burn_override if burn_override is not None else int(em.get("n_burn", 1500))
    n_burn = min(n_burn, max(n_steps // 3, 1))  # never discard more than 2/3 of chain

    init_scale = float(em.get("init_scale", 0.3))
    n_workers = workers_override or int(cfg["n_workers"])

    # pseudo-marginal. seed_day per-theta via seed_day_offset; sim_seed
    # drawn fresh per eval (M particles) unless pm disabled (then frozen).
    shared["_emcee_sim_seed"] = int(cfg["seed"])
    pm = em.get("pseudo_marginal", {})
    shared["pm_enabled"] = bool(pm.get("enabled", False))
    shared["pm_M"] = int(pm.get("n_particles", 1)) if shared["pm_enabled"] else 1
    _pm_M = shared["pm_M"]
    print(
        f"[emcee] R0 prior {cfg['priors']['R0']}; "
        f"f_refer FIXED at {cfg['fixed'].get('f_refer', 0.05):.3f}",
        flush=True,
    )
    print(
        f"[pm] pseudo_marginal={'ON' if shared['pm_enabled'] else 'OFF'}, "
        f"M={_pm_M} particles/eval; seeds drawn fresh per evaluation "
        f"(joint (theta,trajectory) posterior)",
        flush=True,
    )

    chain_h5 = OUT / "emcee_chain.h5"
    backend = _open_or_repair_backend(
        chain_h5, name="mcmc"
    )  # self-heal a spot-reclaim-truncated h5
    resuming = chain_h5.exists() and _backend_iteration(backend) > 0
    n_dim = len(CAL)

    rng = np.random.default_rng(cfg["seed"])
    init: np.ndarray | None = None
    if resuming:
        print(
            f"[emcee] RESUMING from {chain_h5.name}: iteration "
            f"{backend.iteration}/{n_steps}, walkers {backend.shape[0]}",
            flush=True,
        )
        assert backend.shape[0] == n_walkers, (
            f"backend walker count {backend.shape[0]} != requested {n_walkers}; "
            f"delete outputs/emcee_chain.h5 to restart, or match walker count."
        )
    else:
        backend.reset(n_walkers, n_dim)
        if init_npz:
            # Seed from a previous posterior — random walker sample
            prev = np.load(init_npz, allow_pickle=True)
            idx = rng.choice(len(prev[CAL[0]]), size=n_walkers, replace=True)
            init = np.column_stack([prev[k][idx] for k in CAL]).astype(float)
            for j, k in enumerate(CAL):
                s = np.std(prev[k])
                init[:, j] += init_scale * s * rng.standard_normal(n_walkers)
                # Clamp jittered START back inside prior support so no walker
                # begins outside it. Only affects init positions;
                # sampling/scoring/model logic unchanged.
                lo, hi = prior_support(shared["priors"][k])
                init[:, j] = np.clip(init[:, j], lo, hi)
            print(
                f"[emcee] init from {init_npz} ({len(idx)} draws) + jitter (clamped to prior support)",
                flush=True,
            )
        else:
            init = np.zeros((n_walkers, n_dim))
            for w in range(n_walkers):
                for j, k in enumerate(CAL):
                    init[w, j] = sample_prior(shared["priors"][k], rng)
            print(f"[emcee] init from prior for {n_walkers} walkers", flush=True)

    print(
        f"[emcee] {n_walkers} walkers, {n_steps} steps, burn-in {n_burn}, "
        f"{n_workers} workers, n_dim={n_dim}, backend={chain_h5.name}",
        flush=True,
    )

    # run in chunks toward n_steps (30k) so we can (a) stop early once
    # tau has converged, (b) stop on demand via an outputs/STOP sentinel, and
    # (c) snapshot a consistent rotating backup at every chunk boundary for the
    # spot-reclaim h5 repair. Each chunk boundary is a flushed, valid checkpoint.
    chunk = int(em.get("chunk_steps", 1000))
    conv_mult = float(em.get("conv_tau_mult", 50.0))  # converged when iter > mult * max(tau)
    backup_keep = int(em.get("backup_keep", 3))
    stop_file = OUT / "STOP"
    tau_prev = None
    tmax = np.nan  # latest max autocorrelation time; nan until an estimate succeeds
    done_reason = "reached n_steps"
    with Pool(processes=n_workers, initializer=init_worker, initargs=(shared,)) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, _emcee_log_prob, pool=pool, backend=backend
        )
        # When resuming, pass None so emcee reads state from the backend.
        state = init if not resuming else None
        while _backend_iteration(backend) < n_steps:
            this_chunk = min(chunk, n_steps - _backend_iteration(backend))
            sampler.run_mcmc(state, this_chunk, progress=True)
            state = None  # later chunks resume from the backend
            it = _backend_iteration(backend)
            _rotate_backup(chain_h5, keep=backup_keep)  # consistent snapshot post-flush
            try:
                tau_now = sampler.get_autocorr_time(tol=0)
                tmax = float(np.max(tau_now))
                acc_now = float(np.mean(sampler.acceptance_fraction))
                converged = it > conv_mult * tmax
                if tau_prev is not None:
                    converged = converged and bool(
                        np.all(np.abs(tau_now - tau_prev) / tau_now < 0.01)
                    )
                tau_prev = tau_now
                print(
                    f"[emcee] chunk -> iter {it}/{n_steps}  acc {acc_now:.4f}  "
                    f"tau_max {tmax:.1f}  need>{conv_mult * tmax:.0f}  "
                    f"{'CONVERGED' if converged else 'running'}",
                    flush=True,
                )
            except emcee.autocorr.AutocorrError:
                converged = False
                print(f"[emcee] chunk -> iter {it}/{n_steps}  (tau not yet estimable)", flush=True)
            # small READ-SAFE intermediate checkpoint for live plotting. Writes a
            # thinned posterior (theta + selected sim_seed) to a separate few-MB .npz each
            # chunk, so figures can be made from THIS file without pulling/opening the ~GB
            # emcee_chain.h5, which is UNSAFE to read while writing (the HDF5 lock
            # collision can kill a run outright).
            # Atomic write (tmp -> rename) so a reader never sees a half-written file.
            try:
                n_burn_ck = min(n_burn, max(it // 3, 1))
                # THIN VIA emcee's `thin=`, never by striding the flattened chain. The flat
                # layout is row = step * nwalkers + walker, so any stride sharing a factor
                # with nwalkers aliases onto a handful of walkers and stays there (5 of 480
                # at iter 15k; 2 of 480 at 30k). Statistics do not merely narrow under that
                # bug, they FLIP -- a +0.45 parameter correlation reads as -0.12 once
                # corrected. `thin=` strides the STEP axis and keeps every walker.
                kept = max(it - n_burn_ck, 1)
                thin = max(1, (kept * n_walkers) // 50000)  # cap ~50k draws -> small file
                cc = sampler.get_chain(discard=n_burn_ck, thin=thin, flat=True)
                cb = sampler.get_blobs(discard=n_burn_ck, thin=thin, flat=True)
                assert cc is not None and cb is not None  # chunks have run; log-prob returns blobs
                Mck = cb.shape[1] // 2
                sel = select_within_theta(
                    cb[:, Mck:].astype(np.float64), np.random.default_rng(int(cfg["seed"]) + 7)
                )
                ck = {k: cc[:, j] for j, k in enumerate(CAL)}
                ck["seed_day"] = np.round(shared["start_mean"] + ck["seed_day_offset"]).astype(int)
                ck["sim_seed"] = (
                    cb[:, :Mck].astype(np.int64)[np.arange(cc.shape[0]), sel].astype(int)
                )
                ck["_iter"] = np.array([it])
                ck["_tau_max"] = np.array([tmax])
                # np.savez_compressed appends '.npz' if the name lacks it, so write
                # to a '.tmp.npz' base (np leaves it, since it ends in .npz) then rename that
                # exact file. (A '.npz.tmp' name makes np write '.npz.tmp.npz', so the rename
                # target never existed, every checkpoint skipped harmlessly.)
                tmp = OUT / "checkpoint_ppc.tmp.npz"
                np.savez_compressed(tmp, **ck)
                tmp.replace(OUT / "checkpoint_ppc.npz")
            except Exception as _ck_e:  # noqa: BLE001 -- best-effort; must never stop the sampler
                print(f"[emcee] checkpoint skip @ iter {it}: {_ck_e}", flush=True)
            if converged:
                done_reason = "early-stop: tau converged"
                break
            if stop_file.exists():
                done_reason = "early-stop: outputs/STOP sentinel"
                print(f"[emcee] STOP sentinel found — finalizing at iter {it}", flush=True)
                break
    print(
        f"[emcee] sampling finished ({done_reason}) at iter {_backend_iteration(backend)}",
        flush=True,
    )

    # Post-processing. Clamp burn-in to the ACTUAL chain length (early-stop safe).
    n_burn_eff = min(n_burn, max(_backend_iteration(backend) // 3, 1))
    chain = sampler.get_chain(discard=n_burn_eff, flat=True)  # (N, n_dim)
    assert chain is not None  # sampling has run, so emcee has stored steps
    acc = sampler.acceptance_fraction.mean()
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_str = f"{np.mean(tau):.1f} (mean)  {np.max(tau):.1f} (max)"
    except emcee.autocorr.AutocorrError as e:
        tau = None
        tau_str = f"chain too short: {e}"
    print(f"[emcee] mean acceptance {acc:.3f};  integrated autocorr time {tau_str}", flush=True)

    # Save posterior in SIR-compatible layout
    post = {k: chain[:, j] for j, k in enumerate(CAL)}
    N_post = chain.shape[0]
    # seed_day per-sample from seed_day_offset. If f_refer was fixed, blank it.
    post["seed_day"] = np.round(shared["start_mean"] + post["seed_day_offset"]).astype(int)
    if shared.get("pm_enabled", False):
        # recover the M seeds + per-particle logL from emcee blobs, then
        # collapse to a burnout-free per-draw sim_seed by within-theta selection.
        blobs = sampler.get_blobs(discard=n_burn_eff, flat=True)  # (N, 2M) float64
        assert blobs is not None  # _emcee_log_prob always returns the seed/logL blob
        M = int(shared["pm_M"])
        pm_seeds = blobs[:, :M].astype(np.int64)  # (N, M)
        pm_logL = blobs[:, M:].astype(np.float64)  # (N, M)
        post["pm_seeds"] = pm_seeds
        post["pm_logL"] = pm_logL
        sel_rng = np.random.default_rng(int(cfg["seed"]) + 7)
        sel = select_within_theta(pm_logL, sel_rng)
        post["sim_seed"] = pm_seeds[np.arange(N_post), sel].astype(int)
        BURN = -50000.0
        frac_particles = float((pm_logL < BURN).mean())
        frac_selected = float((pm_logL[np.arange(N_post), sel] < BURN).mean())
        print(
            f"[pm] burnout fraction in stored particles: {frac_particles:.3f}; "
            f"after within-theta selection: {frac_selected:.3f} "
            f"(target ~0 — burnouts excluded by construction)",
            flush=True,
        )
    else:
        post["sim_seed"] = np.full(N_post, shared["_emcee_sim_seed"], dtype=int)
    post["zones"] = np.array(ctx["zones"])
    # f_refer is fixed via cfg; write into posterior for downstream compat.
    post["f_refer"] = np.full(N_post, float(cfg["fixed"].get("f_refer", 0.05)), dtype=float)
    np.savez_compressed(OUT / "posterior.npz", **post)

    # Report
    ess_approx = float(N_post / np.mean(tau)) if tau is not None else float("nan")
    report = {
        "sampler": "emcee",
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "n_burn": n_burn,
        "n_burn_effective": int(n_burn_eff),
        "iterations_run": _backend_iteration(backend),
        "stop_reason": done_reason,
        "betabin_s": float(shared["betabin_s"]),
        "N_posterior_draws": int(N_post),
        "acceptance_mean": float(acc),
        "autocorr_mean": float(np.mean(tau)) if tau is not None else None,
        "autocorr_max": float(np.max(tau)) if tau is not None else None,
        "ess_approx": ess_approx,
        "medians": {k: float(np.median(post[k])) for k in CAL},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"[emcee] posterior saved: N={N_post}, ESS~{ess_approx:.0f}", flush=True)


def main(n_override=None, w_override=None, posterior_only=False) -> None:
    ensure_dirs()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    N = n_override or cfg["n_draws"]
    nw = w_override or cfg["n_workers"]
    rng = np.random.default_rng(cfg["seed"])
    ctx = build_context(cfg)
    shared = ctx["shared"]
    Z = ctx["Z"]
    if posterior_only:
        loaded = np.load(OUT / "posterior.npz", allow_pickle=True)
        post = {k: loaded[k] for k in loaded.files}
        rep = json.loads((OUT / "report.json").read_text())
        ess_full = float(rep.get("ESS", rep.get("ess_approx", float("nan"))))
        N = int(rep.get("N", rep.get("N_posterior_draws", 0)))
        print(f"[posterior-only] loaded {N} resampled draws, ESS={ess_full:.1f}")
    else:
        base = int(cfg["seed"])
        tasks = [(i, base + i, base + 7_000_000 + i) for i in range(N)]
        logL = np.zeros(N)
        pars = {k: np.zeros(N) for k in CAL}
        seed_days = np.zeros(N, int)
        sim_seeds = np.array([t[2] for t in tasks])
        print(
            f"{Z} zones; mobility={cfg.get('mobility_source')}; horizon={cfg['horizon_date']}; "
            f"confirmed step={cfg['confirmed_step_days']}d -> {len(shared['obs_conf_new'])} pts; "
            f"backlog targets {ctx['shared']['obs_backlog'].tolist()}; scoring {N} draws on {nw} workers..."
        )
        with ProcessPoolExecutor(
            max_workers=nw, initializer=init_worker, initargs=(shared,)
        ) as ex:
            for i, lL, vals, sd in ex.map(score_draw, tasks, chunksize=max(1, N // (nw * 8))):
                logL[i] = lL
                seed_days[i] = sd
                for k, v in zip(CAL, vals):
                    pars[k][i] = v
        ess_full, w = ess_of(logL)
        rs_idx = rng.choice(N, size=N, p=w)
        print(f"ESS(N={N}) = {ess_full:.1f}  ESS/N = {ess_full / N:.4f}")

        post = {k: pars[k][rs_idx] for k in CAL}
        post["seed_day"] = seed_days[rs_idx]
        post["sim_seed"] = sim_seeds[rs_idx]
        post["zones"] = np.array(ctx["zones"])
        np.savez_compressed(OUT / "posterior.npz", **post)

    # PPC = TWO sets: prior predictive (uniform subset of all N draws)
    # AND posterior predictive (importance-resampled). Shows what ABC inherited vs what it
    # constrained — useful when ESS is small. At smoke N=80 we use all draws for prior.
    n_ppc = min(600, N)
    if posterior_only:
        # No unresampled `pars` available — use the resampled posterior for both
        # prior and posterior slots (prior context unavailable in this mode).
        idx_prior = rng.choice(len(post["R0"]), size=n_ppc, replace=True)
        idx_post = rng.choice(len(post["R0"]), size=n_ppc, replace=True)
        all_tasks = [
            (tuple(post[k][j] for k in CAL), int(post["seed_day"][j]), int(post["sim_seed"][j]))
            for j in np.concatenate([idx_prior, idx_post])
        ]
    else:
        prior_sel = rng.choice(N, size=n_ppc, replace=(n_ppc > N))  # uniform over all draws
        post_sel = rng.choice(rs_idx, size=n_ppc, replace=True)  # importance-resampled
        all_sel = np.concatenate([prior_sel, post_sel])
        all_tasks = [(tuple(pars[k][j] for k in CAL), seed_days[j], sim_seeds[j]) for j in all_sel]

    # drop out-of-support tasks BEFORE mapping (mirrors sampling; keeps the
    # prior/posterior block boundary at n_ppc intact by shrinking both halves equally
    # is not possible per-half, so re-pick n_ppc as the min in-support count per half).
    # NB: this filter runs in the MAIN process, where the module global `SHARED` is not set
    # (only workers get it via init_worker), so use the local `shared["priors"]` explicitly.
    def _in_support_task(t):
        return all(
            np.isfinite(log_prior(shared["priors"][k], float(v))) for k, v in zip(CAL, t[0])
        )

    prior_tasks = [t for t in all_tasks[:n_ppc] if _in_support_task(t)]
    post_tasks = [t for t in all_tasks[n_ppc:] if _in_support_task(t)]
    n_ppc_eff = min(len(prior_tasks), len(post_tasks))
    if n_ppc_eff < n_ppc:
        print(
            f"[ppc] {n_ppc - n_ppc_eff} out-of-support draws dropped per half; "
            f"PPC uses n={n_ppc_eff} (was {n_ppc})",
            flush=True,
        )
    all_tasks = prior_tasks[:n_ppc_eff] + post_tasks[:n_ppc_eff]
    n_ppc = n_ppc_eff
    with ProcessPoolExecutor(max_workers=nw, initializer=init_worker, initargs=(shared,)) as ex:
        ppc_all = list(ex.map(ppc_draw, all_tasks, chunksize=max(1, len(all_tasks) // (nw * 4))))

    def split(idx):
        prior = np.array([ppc_all[k][idx] for k in range(n_ppc)])
        post = np.array([ppc_all[k][idx] for k in range(n_ppc, 2 * n_ppc)])
        return prior, post

    nat_prior, nat_pp = split(0)
    lat_prior, lat_pp = split(1)
    conf_new_prior, conf_new_pp = split(2)
    confirmed_iso_prior, confirmed_iso_pp = split(3)
    suspect_iso_prior, suspect_iso_pp = split(4)
    iso_stock_prior, iso_stock_pp = split(5)
    queue_prior, queue_pp = split(6)
    zone_final_prior, zone_final_pp = split(7)
    deaths_binned_prior, deaths_binned_pp = split(8)
    adm_prior, adm_pp = split(9)
    rec_prior, rec_pp = split(10)
    _, zone_deaths_final_pp = split(11)

    fig_all_in_one(
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
        ess_full,
        N,
        prior_arrays={
            "nat": nat_prior,
            "lat": lat_prior,
            "conf": conf_new_prior,
            "deaths": deaths_binned_prior,
            "zone": zone_final_prior,
            "queue": queue_prior,
            "conf_iso": confirmed_iso_prior,
            "susp_iso": suspect_iso_prior,
            "iso_stock": iso_stock_prior,
            "adm": adm_prior,
            "rec": rec_prior,
        },
    )
    fig_scored_only(
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
        ess_full,
        N,
    )
    fig_likelihood_diagnosis(
        ctx,
        conf_new_pp,
        deaths_binned_pp,
        confirmed_iso_pp,
        zone_final_pp,
        zone_deaths_final_pp,
        lat_pp,
        ess_full,
        N,
    )
    fig_ppc(ctx, nat_pp, lat_pp, conf_new_pp, post, ess_full, N)
    fig_capacity_iso(ctx, queue_pp, confirmed_iso_pp, suspect_iso_pp, iso_stock_pp, post, N)
    fig_spatial(ctx, zone_final_pp)
    fig_posterior(post)

    bk_med = np.median(queue_pp, axis=0).tolist()
    report = {
        "N": int(N),
        "ESS": float(ess_full),
        "ESS_over_N": float(ess_full / N),
        "medians": {k: float(np.median(post[k])) for k in CAL},
        # R_floor RE-OPENED to CAL — report its posterior quantiles (was a fixed
        # value when it is closed). hill_n stays FIXED.
        "R_floor_q": {q: float(np.percentile(post["R_floor"], q)) for q in (5, 50, 95)},
        "cum_half_q": {q: float(np.percentile(post["cum_half"], q)) for q in (5, 50, 95)},
        "hill_n_fixed": float(cfg["fixed"]["hill_n"]),
        "sq_trace_eff_q": {q: float(np.percentile(post["sq_trace_eff"], q)) for q in (5, 50, 95)},
        "obs_cum_end": float(ctx["obs_nat_cum"][-1]),
        "ppc_cum_end_median": float(np.median(nat_pp[:, -1])),
        "latent_burden_end_q": {q: float(np.percentile(lat_pp[:, -1], q)) for q in (5, 50, 95)},
        "obs_backlog": ctx["shared"]["obs_backlog"].tolist(),
        "model_backlog_median": bk_med,
        "obs_confirmed_iso_range": [
            int(ctx["shared"]["obs_confirmed_iso"].min()),
            int(ctx["shared"]["obs_confirmed_iso"].max()),
        ],
        "model_confirmed_iso_end_median": float(np.median(confirmed_iso_pp[:, -1])),
        "model_total_iso_end_median": float(np.median(iso_stock_pp[:, -1])),
        "obs_adm_total": float(ctx["shared"]["obs_adm_new"].sum()),
        "model_adm_total_median": float(np.median(adm_pp.sum(axis=1))),
        "obs_rec_total": float(ctx["shared"]["obs_rec_new"].sum()),
        "model_rec_total_median": float(np.median(rec_pp.sum(axis=1))),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
