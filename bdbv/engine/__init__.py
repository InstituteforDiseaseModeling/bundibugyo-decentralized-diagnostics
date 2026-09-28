"""Calibration and simulation engine: likelihood, samplers, and figures.

Builds the shared 470-zone calibration context (cases, deaths, mobility,
lab rollout, care referral), scores a parameter vector against the
observed data, and drives either prior-predictive SIR or emcee ensemble
MCMC. Entry points are `main()` and `main_emcee()`, both reached through
`calibration/run.py`.

The package is split by concern and this module re-exports the names used
elsewhere, so `from bdbv import engine as E; E.CAL, E.build_context, ...` works:

    state.py       artefact paths, CAL / FIXED_FROM_CAL, brake reforms, `SHARED`, `init_worker`
    context.py     `build_context`: observed data, mobility, lab rollout -> the worker context
    simulation.py  `make_params`, `simulate_draw`, observation summaries (`expected_summaries`)
    scoring.py     `log_likelihood`, SIR/PPC worker tasks, pseudo-marginal seed selection
    samplers.py    `main_emcee` and `main` (SIR), HDF5 backend backup/repair
    figures.py     calibration diagnostic figures

Run:  uv run python calibration/run.py [--sampler emcee] [--workers W]
"""

from __future__ import annotations

from .context import build_context, build_zone_set
from .samplers import main, main_emcee
from .scoring import draw_trajectories, ess_of, weighted_band
from .simulation import make_params, three_tier_mix, zone_r_multiplier
from .state import CAL, CORE, FIG, FIXED_FROM_CAL, HERE, OUT, REFORM_PARAMS, init_worker

# The names calibration/ and scenarios/ reach through `bdbv.engine`.
__all__ = [
    "CAL",
    "CORE",
    "FIG",
    "FIXED_FROM_CAL",
    "HERE",
    "OUT",
    "REFORM_PARAMS",
    "build_context",
    "build_zone_set",
    "draw_trajectories",
    "ess_of",
    "init_worker",
    "main",
    "main_emcee",
    "make_params",
    "three_tier_mix",
    "weighted_band",
    "zone_r_multiplier",
]
