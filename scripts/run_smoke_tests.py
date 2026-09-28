"""Run small end-to-end checks for calibration, scenario, and figure code."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    merged_env = {**os.environ, **(env or {})}
    merged_env.setdefault("BDBV_DATA_DIR", str(ROOT / "data" / "BDBV2026-Data"))
    print(f"\n[{cwd.relative_to(ROOT)}] + {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=merged_env)


def require_data() -> None:
    required = [
        ROOT / "data" / "BDBV2026-Data" / "data" / "insp_sitrep" / "processed",
        ROOT / "data" / "BDBV2026-Data" / "data" / "osrm" / "processed",
        ROOT
        / "data"
        / "flowminder_hdx"
        / "processed"
        / "flowminder_hdx__outflow_2026_03.matrix.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing staged data:\n  - "
            + "\n  - ".join(missing)
            + "\nRun `python scripts/stage_data.py` first."
        )


def main() -> None:
    require_data()

    run(
        [
            sys.executable,
            "run.py",
            "--sampler",
            "emcee",
            "--steps",
            "1",
            "--walkers",
            "28",
            "--workers",
            "2",
            "--burn",
            "0",
        ],
        cwd=ROOT / "calibration",
    )

    checkpoint = ROOT / "data" / "inputs" / "source_checkpoint_iter44000.npz"
    dest = ROOT / "scenarios" / "outputs" / checkpoint.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dest)

    run([sys.executable, "extract_subsample.py", "--n", "3"], cwd=ROOT / "scenarios")
    run(
        [
            sys.executable,
            "run.py",
            "--draws",
            "1",
            "--conts",
            "1",
            "--workers",
            "1",
            "--streams",
            "base,time,place,timerdt",
            "--fresh",
        ],
        cwd=ROOT / "scenarios",
    )
    run([sys.executable, "scripts/make_figures.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
