"""Run-wide state shared by the engine modules.

Artefact paths (derived from the invoking script's location), the calibrated/fixed parameter
lists and the brake-reform switch that edits them in place, the per-process shared context
`SHARED` that worker initialisers fill, and the per-worker pseudo-marginal RNG.
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path

import numpy as np
import yaml

from ..model.priors import log_prior

# Artefact locations derive from argv[0], so the engine writes alongside the entry point
# that invoked it (`calibration/run.py` -> `calibration/outputs/`). The directories are
# created by the entry points in samplers.py, NOT here: importing this module should not leave
# `figures/` and `outputs/` behind in whatever directory the importer happened to run from.
HERE = Path(_sys.argv[0]).resolve().parent
FIG, OUT = HERE / "figures", HERE / "outputs"


def ensure_dirs() -> None:
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)


def prior_support(spec: dict) -> tuple[float, float]:
    """(lo, hi) support of a prior spec, for clamping jittered emcee init back inside
    the prior. `--init-npz` jitter (init_scale*std) can push a walker's START
    outside prior support, e.g. short_trips_alpha (uniform 0..1) to -0.17/1.07.
    Walkers should never START outside the prior anyway."""
    d = spec.get("dist")
    eps = 1e-6
    if d == "uniform":
        return float(spec["min"]), float(spec["max"])
    if d == "beta":
        return eps, 1.0 - eps  # open (0,1)
    if d == "normal":
        return float(spec.get("min", -np.inf)), float(spec.get("max", np.inf))
    if d == "lognormal":
        return eps, np.inf
    return -np.inf, np.inf


def theta_out_of_support(theta) -> bool:
    """True if any CAL value in `theta` falls outside its prior support.

    The PPC re-simulation path (`ppc_draw`) must SKIP out-of-support draws,
    exactly as the SAMPLING path (`_emcee_log_prob`, which returns -inf on a
    non-finite log_prior before ever simulating) already does. A saved chain can
    contain ~2-5% out-of-support draws; they get -inf posterior mass so are
    harmless to inference, but they must not contribute to PPCs."""
    for k, v in zip(CAL, theta):
        if not np.isfinite(log_prior(SHARED["priors"][k], float(v))):
            return True
    return False


CORE = ["Bunia", "Mongbwalu", "Rwampara", "Nyankunde"]  # canonical spellings
CAL = [
    "R0",
    "jump_prob",
    "ascertainment",
    "ifr",
    "sq_trace_eff",
    "p_iso_presump",
    "cum_half",  # Hill brake half-point
    # pipeline knobs. `p_iso_confirmed` is NOT sampled: `iso_uplift` is, and the two tiers
    # are nested — see make_params.
    "iso_uplift",
    "collect_ratio",
    "short_trips_alpha",  # tier-1 short-trips mix
    "cadaver_swab_coverage",  # post-mortem PCR reach
    "seed_day_offset",  # continuous seed-day offset from start_date_mean
    "rho_access",  # R multiplier for peaceful / better-served zones
]
# `nat_half` and `nat_floor` are NOT in CAL. Under `nat_brake_placement: global` the
# deaths-driven gap-brake they parameterise is replaced, not supplemented, by the S_nat clock —
# model.py's __post_init__ refuses a config that carries both. They stay in FIXED_FROM_CAL at
# their 0.0 no-ops so every downstream `p["nat_half"]` lookup is untouched. R_floor is opened
# instead, via `brake_reforms: [R_floor]`.

# --- the three brake-form reforms, switchable without editing this file ----------------------
#
# Each reform is "free a parameter that the current form pins by fiat". Which of them are OPEN is
# a property of the run, not of the code, so that they can be swept singly and in pairs. The
# CAL/FIXED split for these three is therefore read from
# `config.yaml` — a reform is CAL if its name appears under `params:`, FIXED if under `fixed:` —
# with the `BRAKE_REFORMS` env var overriding for sweep scripts (comma-separated, "" = none).
#
# This is the ONE place the reform set is decided. Everything downstream (`make_params`, the
# emcee log-prob, the posterior writer) reads `p[name]` and is indifferent.
REFORM_PARAMS = [
    "hill_n",  # A: step softening   — Hill exponent, otherwise fixed at 8.7
    #    because it was unidentified; aggregation argues for 1-2
    "R_floor",  # B: cold asymptote   — fixed 0.80; sole controller of late core
    #    r_eff once B_loc ~ 0, last fitted under a prior with support (0,1)
    "tau_w",  # C: waning memory    — driver decay time; inf = current form
]


def _open_reforms(cfg: dict | None = None) -> list[str]:
    """Which of REFORM_PARAMS this run calibrates: `brake_reforms:` in config.yaml, or the
    BRAKE_REFORMS env var (comma-separated; the empty string means none). See REFORM_PARAMS."""
    want = os.environ.get("BRAKE_REFORMS")
    if want is None:
        config = cfg if cfg is not None else yaml.safe_load((HERE / "config.yaml").read_text())
        want = (config or {}).get("brake_reforms") or []
    else:
        want = [s.strip() for s in want.split(",") if s.strip()]
    bad = set(want) - set(REFORM_PARAMS)
    if bad:
        raise ValueError(
            f"brake_reforms names non-reform params {sorted(bad)}; "
            f"expected a subset of {REFORM_PARAMS}"
        )
    return [r for r in REFORM_PARAMS if r in want]  # canonical order, not the caller's


def _apply_reforms(open_reforms: list[str]) -> None:
    """Move the open reforms from FIXED_FROM_CAL into CAL, in place.

    Mutation rather than rebinding: worker processes import this module fresh and re-run this at
    init, and every other module holds `run.CAL` by attribute lookup — but `_emcee_log_prob` and
    the gate scripts zip CAL against a theta vector, so the two lists must never disagree between
    parent and worker. `init_worker` re-applies from the same env/config the parent used, so they cannot.
    """
    for r in REFORM_PARAMS:
        want_cal = r in open_reforms
        if want_cal and r not in CAL:
            CAL.append(r)
        if not want_cal and r in CAL:
            CAL.remove(r)
        if want_cal and r in FIXED_FROM_CAL:
            FIXED_FROM_CAL.remove(r)
        if not want_cal and r not in FIXED_FROM_CAL:
            FIXED_FROM_CAL.append(r)


# k, R_floor, hill_n, p_clin, cap_phei, noso_boost are held FIXED (see
# config.yaml `fixed` block for rationale). FIXED_FROM_CAL merges their values
# into `p` at construction time so every downstream p["name"] lookup in
# make_params etc. is untouched — only CAL (what emcee samples) shrank.
# nat_half + nat_floor are here too (both 0.0 in `fixed:` — the gap-brake is off under `global`).
FIXED_FROM_CAL = [
    "k",
    "hill_n",
    "p_clin",
    "cap_phei",
    "noso_boost",
    "R_floor",
    "nat_half",
    "nat_floor",
]
APPLIED_REFORMS = _open_reforms()  # decided once, at import, from config/env
_apply_reforms(APPLIED_REFORMS)


SHARED: dict = {}
_PM_RNG: np.random.Generator | None = None  # per-worker RNG for pseudo-marginal seeds


def init_worker(shared):
    SHARED.update(shared)
    # Re-apply the reform set inside the worker. A spawn-started worker re-imports this module
    # and re-runs `_open_reforms()` from the same config file, so CAL already matches — but a
    # sweep script that sets BRAKE_REFORMS *after* import would leave parent and worker
    # disagreeing, and CAL is zipped against theta vectors positionally. Reading
    # it again here makes the worker's CAL a function of the environment it actually runs in.
    _apply_reforms(_open_reforms())
    # Fresh per-worker entropy so each process draws independent
    # simulator seeds for pseudo-marginal scoring (see _emcee_log_prob).
    global _PM_RNG
    _PM_RNG = np.random.default_rng()


def pm_rng():
    global _PM_RNG
    if _PM_RNG is None:
        _PM_RNG = np.random.default_rng()
    return _PM_RNG
