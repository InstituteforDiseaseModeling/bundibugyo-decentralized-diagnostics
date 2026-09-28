# Decentralized diagnostics as a spatial control strategy for the 2026 Bundibugyo Ebola epidemic

Code and data for the preprint:

> Decentralized diagnostics as a spatial control strategy for the 2026
> Bundibugyo Ebola epidemic

It contains the calibration and intervention code needed to rerun the main
analyses, plus compact frozen summaries that regenerate the manuscript figures
without re-running a full MCMC chain or scenario sweep.

## Install

Clone the repository and sync the locked Python environment:

```bash
git clone https://github.com/InstituteforDiseaseModeling/bundibugyo-decentralized-diagnostics.git
cd bundibugyo-decentralized-diagnostics
uv sync --frozen
```

The project requires Python 3.12 or newer (`.python-version` pins 3.13) and is
managed with `uv`. See the uv installation guide at <https://docs.astral.sh/uv/getting-started/installation/>
if `uv` is not already installed.

## Contents

- `data/`: staged public data, curated model inputs, and frozen figure inputs;
  see `data/DATA.md`.
- `bdbv/`: the transmission model (`bdbv/model/`) and the calibration engine
  (`bdbv/engine/`: shared state, calibration context, simulation wrapper,
  likelihood scoring, samplers, and diagnostic figures).
- `calibration/`: the calibration configuration and sampler entry point.
- `scenarios/`: the intervention sweep and postprocessing scripts.
- `scripts/stage_data.py`: stages public INRB/UMIE and Flowminder HDX inputs
  under `data/`.
- `scripts/prepare_scenario_checkpoint.py`: copies a calibration checkpoint
  into the filename expected by the scenario sweep.
- `scripts/run_smoke_tests.py`: runs a tiny calibration and scenario sweep.
- `scripts/make_figures.py`: a standalone script that regenerates all preprint
  result figures from `data/figure_inputs/`.

## Stage the public inputs

```bash
uv run python scripts/stage_data.py
```

This clones the public INRB/UMIE data repository, checks out the pinned commit
used for the manuscript analyses, downloads the Flowminder HDX mobility CSVs,
and writes the processed March 2026 origin-destination matrix used by the
model.

## Smoke test

```bash
uv run python scripts/run_smoke_tests.py
```

The smoke test runs a one-step `emcee` calibration, extracts a three-draw
scenario subsample, runs one intervention draw, and rebuilds the frozen
manuscript figures.

## Checks

The unit tests and lint checks run in CI and need no staged data:

```bash
uv run python -m unittest discover -s tests
uvx ruff@0.16.9 check .
uvx ruff@0.16.9 format --check .
```

## Rebuild the figures

Generate the figures with:

```bash
uv run python scripts/make_figures.py
```

The script writes PNG files to `figures/`.

## Rerun analyses

The calibration and the scenario sweep each write their outputs next to the
`run.py` that was invoked, so these commands change directory as they go.
Starting from the repository root:

```bash
cd calibration
uv run python run.py --sampler emcee

cd ..
uv run python scripts/prepare_scenario_checkpoint.py

cd scenarios
uv run python extract_subsample.py
uv run python run.py --streams base,time,place,timerdt
uv run python run.py --streams place --tag d30 --rep-delay 30
uv run python analyze.py
uv run python avertable.py
uv run python time_avertable.py outputs/results.jsonl outputs/time_avertable.json time
uv run python time_avertable.py outputs/results.jsonl outputs/timerdt_avertable.json timerdt
uv run python place_avertable.py \
  outputs/results_d30.jsonl outputs/place_avertable_d30_pct_deaths.json pct_deaths
```

The second sweep re-runs the placement stream with labs deployed 30 days after
the cut-off (the configured default is 14 days); the placement figure reads
that sweep, and `--tag d30` keeps its outputs separate from the main sweep's.

### Reproducing the published results exactly

The manuscript results come from the calibration checkpoint bundled as
`data/inputs/source_checkpoint_iter44000.npz`. To reproduce them exactly, skip
the calibration and prepare the scenario sweep from that file, then run the
`scenarios/` commands above:

```bash
uv run python scripts/prepare_scenario_checkpoint.py \
  --source data/inputs/source_checkpoint_iter44000.npz
```

Every step from there is seeded from `scenarios/config.yaml`. A fresh
`calibration/run.py` chain is statistically equivalent but not bit-identical:
in this release neither emcee's proposal stream nor the pseudo-marginal
simulator seeds are derived from the configured seed.

The staging, smoke-test, and figure commands above are all run from the
repository root. `prepare_scenario_checkpoint.py` preserves the calibration
iteration in the copied filename. If it writes a filename other than
`source_checkpoint_iter44000.npz`, update `intervention.posterior` in
`scenarios/config.yaml` to that path before extracting the scenario subsample.

## Frozen inputs

`data/figure_inputs/` holds compact exports of the calibration posterior and the
intervention sweep, so `scripts/make_figures.py` reproduces every manuscript
figure without first re-running the full chain and sweep. The files correspond to
the outputs the full pipeline writes to `calibration/outputs/` and
`scenarios/outputs/`:

- `sq_draws.npz`: the held-out posterior draws used to drive the forward
  scenarios.
- `avertable.json`, `time_avertable.json`, `timerdt_avertable.json`,
  `place_avertable_d30_pct_deaths.json`: averted-burden summaries produced by
  `scenarios/avertable.py`, `time_avertable.py`, and `place_avertable.py`.

Three bulky or code-dependent products are frozen into smaller derived files:

- `status_quo_540.csv` aggregates the ~800 MB `results.jsonl` row stream to one
  status quo endpoint row per posterior draw.
- `posterior_predictive.npz` freezes already simulated posterior predictive
  trajectories for the appendix model-fit panel.
- `placement_map.geojson` freezes the illustrative 20-lab, 30-day realization
  used in the map panel as per-zone burdens plus existing/new lab flags.

Those exports also make the figure build independent of the processed sitrep,
Flowminder, and OSRM inputs, so the figures can be rebuilt without staging data.

## Citation

If you use this code, please cite the preprint. The citation will be added here
once it is posted.

Analyses in this repository depend on public INSP/CUOSP situation reports and
three external data sources, which should
be cited alongside it:

- Outbreak situation reports: INSP/CUOSP, MVE-17 SitReps posted at
  <https://insp.cd/category/sitrep/>.
- Outbreak surveillance data: INRB/UMIE, `BDBV2026-Data`,
  <https://github.com/INRB-UMIE/BDBV2026-Data>. Its health-zone geometries
  derive from GRID3 COD Health Zones v8.0, licensed CC BY 4.0.
- Mobility data: Flowminder Foundation, estimated relocations and residents for
  the Democratic Republic of the Congo, published on HDX
  (<https://data.humdata.org>).
- Conflict-event data: ACLED (Armed Conflict Location & Event Data), accessed
  5 August 2026, province-level DRC event rates transformed by the authors into
  the normalized `s_disruption` accessibility score, <https://www.acleddata.com>.
  Cite Raleigh, Kishi, and Linke (2023), Humanities and Social Sciences
  Communications, DOI `10.1057/s41599-023-01559-4`.

## License

Released under the MIT License; see `LICENSE`. The license covers the code and
the synthetic model outputs in this repository. Curated or transformed inputs
derived from INRB/UMIE SitReps, Flowminder HDX mobility, ACLED data, and GRID3
COD Health Zones v8.0 retain their upstream terms; this includes the curated
files under `data/inputs/`, the `data/figure_inputs/*` files that retain
SitRep-derived observed arrays or GRID3 geometry, and the transformed
`bdbv/province_covariates*` files. See `data/DATA.md` for per-file provenance.
