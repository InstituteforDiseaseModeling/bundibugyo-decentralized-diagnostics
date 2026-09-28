"""Calibration entry point.

Thin CLI wrapper over `bdbv.engine`. Kept as a script in this directory
because the engine derives its `figures/` and `outputs/` locations from
`sys.argv[0]`, so running it from here is what puts artefacts alongside
`calibration/config.yaml`.

The engine is imported as a normal module (not executed via `runpy` as a
synthetic `__main__`) so that the worker functions it hands to
`ProcessPoolExecutor` remain picklable under the macOS spawn start method.

Run:  uv run python calibration/run.py --sampler emcee
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdbv import engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sampler",
        choices=["sir", "emcee"],
        default="sir",
        help="SIR (default, prior-predictive) or emcee ensemble MCMC.",
    )
    ap.add_argument(
        "--draws", type=int, default=None, help="SIR only: number of prior draws to score."
    )
    ap.add_argument(
        "--steps", type=int, default=None, help="emcee only: number of MCMC steps per walker."
    )
    ap.add_argument("--walkers", type=int, default=None, help="emcee only: number of walkers.")
    ap.add_argument(
        "--workers", type=int, default=None, help="Number of parallel worker processes."
    )
    ap.add_argument(
        "--init-npz",
        type=str,
        default=None,
        help="emcee only: initialize walkers from this posterior.npz.",
    )
    ap.add_argument(
        "--burn",
        type=int,
        default=None,
        help="emcee only: override n_burn (auto-clamped to steps//3).",
    )
    ap.add_argument(
        "--posterior-only",
        action="store_true",
        help="SIR only: skip scoring; reuse outputs/posterior.npz for PPC + figs.",
    )
    a = ap.parse_args()

    if a.sampler == "emcee":
        engine.main_emcee(
            steps_override=a.steps,
            walkers_override=a.walkers,
            workers_override=a.workers,
            init_npz=a.init_npz,
            burn_override=a.burn,
        )
    else:
        engine.main(a.draws, a.workers, posterior_only=a.posterior_only)


if __name__ == "__main__":
    main()
