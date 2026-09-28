"""Copy a calibration checkpoint into the scenario input directory.

The scenario sweep is driven by a posterior checkpoint rather than a finished
posterior. Preserve the checkpoint iteration in the output filename so a
60,000-step checkpoint cannot silently masquerade as the manuscript's
44,000-step source.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ITER_RE = re.compile(r"_iter(\d+)\.npz$")


def read_iteration(path: Path) -> int:
    with np.load(path) as checkpoint:
        if "_iter" not in checkpoint:
            raise KeyError(f"{path} does not contain an _iter field")
        return int(np.ravel(checkpoint["_iter"])[0])


def default_dest(iteration: int) -> Path:
    return ROOT / "scenarios" / "outputs" / f"source_checkpoint_iter{iteration}.npz"


def check_dest_iteration(source: Path, dest: Path, iteration: int) -> None:
    match = ITER_RE.search(dest.name)
    if match and int(match.group(1)) != iteration:
        raise ValueError(
            f"{source} contains _iter={iteration}, but {dest.name} names "
            f"iteration {match.group(1)}. Choose {default_dest(iteration)} or "
            "pass --allow-iteration-mismatch if you intentionally want a "
            "non-provenance filename."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "calibration" / "outputs" / "checkpoint_ppc.npz",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=(
            "Destination .npz. Defaults to scenarios/outputs/source_checkpoint_iter{_iter}.npz."
        ),
    )
    parser.add_argument(
        "--allow-iteration-mismatch",
        action="store_true",
        help="Allow an iterNNN destination filename that does not match source _iter.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    try:
        iteration = read_iteration(args.source)
        dest = args.dest or default_dest(iteration)
        if not args.allow_iteration_mismatch:
            check_dest_iteration(args.source, dest, iteration)
    except Exception as err:
        raise SystemExit(f"ERROR: {err}") from err

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, dest)
    print(f"{args.source} -> {dest}")
    if dest.name != "source_checkpoint_iter44000.npz":
        try:
            rel = dest.resolve().relative_to((ROOT / "scenarios").resolve())
        except ValueError:
            rel = dest
        print(
            f"\nNext, point scenarios/config.yaml intervention.posterior at {rel} "
            "before running extract_subsample.py.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
