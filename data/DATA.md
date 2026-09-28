# Data

This directory separates public source data from the compact, tracked artifacts
bundled with the preprint reproduction package.

## Staged external data

Run the staging script from the repository root:

```bash
uv run python scripts/stage_data.py
```

It takes a few minutes and writes about 800 MB into two ignored directories:

- `BDBV2026-Data/`: a clone of the public INRB/UMIE Ebola data repository,
  <https://github.com/INRB-UMIE/BDBV2026-Data>, checked out to the pinned commit
  `60f67015625a9f1cf54d9da5339926121d90b72d`. The model reads only the
  `data/*/processed/` trees; nothing reads `data/*/raw/`.
- `flowminder_hdx/`: Flowminder origin-destination CSVs from HDX, plus the
  processed March 2026 square outflow matrix used by `load_mobility_outflow`.
  The matrix is rebuilt from
  `drc-estimated-relocations-2020_03-2026_04-v2.0-external.csv`, taken from the
  HDX datasets `bd5781f3-9c6a-427a-955b-ce2b59def8c3` and
  `f3660479-fb8d-4a74-baa7-09d2d24c8e0d`.

These directories are not tracked because they are public data products that
can be recreated by `scripts/stage_data.py`.

Both sources are pinned. The INRB/UMIE repository is checked out at a fixed
commit. HDX, unlike a git remote, republishes resources in place, so the
package IDs alone are not a pin: `stage_data.py` additionally verifies the
SHA256 of the downloaded relocation CSV and of the processed outflow matrix it
derives, and aborts if either differs from the version the manuscript was built
from. Pass `--allow-hdx-drift` to stage newer upstream data anyway; that data
will not reproduce the published figures.

Health-zone names in the rebuilt matrix are the raw HDX spellings, which differ
from the canonical zone names for a handful of zones (`Gety`/`Gethy`,
`Nia Nia`/`Nia-Nia`, and similar). `model.data` canonicalizes row and column
names on load and sums any duplicates, so the matrix the model actually
consumes is unaffected by which spelling the source file carries.

## Tracked analysis inputs

The `inputs/` directory contains small curated inputs that the model or smoke
test reads directly:

- `lab_rollout.yaml`: PCR laboratory rollout dates and health-zone locations
  used to reconstruct observed decentralization during calibration.
- `MVE_isolation_stockflow_validation.xlsx`: a reconciled national isolation
  stock-flow series extracted from SitRep PDFs.
- `source_checkpoint_iter44000.npz`: the posterior draws the published forward
  scenarios were run from. This is a thinned `emcee` checkpoint written at
  iteration 44,000 of a 60,000-step chain — 49,920 draws, walker-major (480
  walkers x 104 thinned steps). The calibration continued past this point, but
  the manuscript results were produced from this checkpoint, not from the
  finished chain. The scenario extractor takes 2,500 draws from it and then
  randomly splits them into 500 greedy-placement training rows plus 2,000
  held-out scoring rows; `figure_inputs/sq_draws.npz` contains the 2,000
  scored rows, and the scenario smoke test reuses the same source file.

  Two caveats belong with any number taken from it, and are stated at greater
  length in `scenarios/config.yaml`:

  - *Not stationary.* The chain is roughly 7 autocorrelation times past
    burn-in with acceptance still falling, and `ascertainment` is still rising
    monotonically. Quantities derived from this checkpoint will move if the
    chain is run to completion.
  - *No stored likelihood.* The checkpoint carries theta and the selected
    `sim_seed` only, not `pm_seeds`/`pm_logL`, so a replay cannot be checked
    against a stored value.

## Frozen figure inputs

The `figure_inputs/` directory contains compact exports from the full fitting
and scenario sweeps. They let `scripts/make_figures.py` regenerate the
manuscript figures without committing a full MCMC chain or the full
intervention `results.jsonl` stream.

`placement_map.geojson` carries health-zone geometry plus only the fields the
map panel draws: `nom` (zone name), `in_model`, the per-zone burdens `sq`,
`random` and `celf`, and the lab flags `existing_lab`, `random_lab` and
`celf_lab`. It was exported from a zone feature collection that also carried
SitRep summaries, response narrative, and WorldPop/GDP/CCVI context; those
fields are not used by any figure and have been removed.

## Tracked artifact provenance

| Path | Kind | Immediate source | Upstream source and permission |
| -- | -- | -- | -- |
| `data/inputs/lab_rollout.yaml` | Curated PCR rollout timeline | Extracted by the authors from public French INSP/CUOSP SitReps and the author-curated `MVE17_diagnostics_timetable_vJul102026.xlsx` workbook | Public INSP/CUOSP SitRep-derived data from PDFs posted at <https://insp.cd/category/sitrep/>; no license click-through observed 2026-09-28 |
| `bdbv/model/care_facilities.yaml` | Curated CTE-zone list | Extracted by the authors from the `Care_facilities` sheet of `MVE17_diagnostics_timetable_vJul102026.xlsx` | Public INSP/CUOSP SitRep-derived data from PDFs posted at <https://insp.cd/category/sitrep/>; no license click-through observed 2026-09-28 |
| `data/inputs/MVE_isolation_stockflow_validation.xlsx` | Reconciled national isolation stock-flow series | Author reconciliation of national isolation stocks and flows from SitRep PDFs | Public INSP/CUOSP SitRep-derived data from PDFs posted at <https://insp.cd/category/sitrep/>; no license click-through observed 2026-09-28 |
| `data/inputs/source_checkpoint_iter44000.npz` | Calibration checkpoint fixture | Thinned checkpoint written by `bdbv.engine` at iteration 44,000 of the manuscript calibration chain | Derived model output from the tracked code and the curated/staged inputs |
| `data/figure_inputs/sq_draws.npz` | Scenario posterior subsample | `scenarios/extract_subsample.py` applied to `source_checkpoint_iter44000.npz` | Derived model output |
| `data/figure_inputs/avertable.json` | Frozen averted-burden summary | `scenarios/avertable.py` applied to the full scenario sweep | Derived model output |
| `data/figure_inputs/time_avertable.json` | Frozen deployment-delay summary | `scenarios/time_avertable.py` applied to the full scenario sweep | Derived model output |
| `data/figure_inputs/timerdt_avertable.json` | Frozen deployment-delay/RDT summary | `scenarios/time_avertable.py` applied to the full `timerdt` scenario stream | Derived model output |
| `data/figure_inputs/place_avertable_d30_pct_deaths.json` | Frozen placement summary | `scenarios/place_avertable.py` applied to the 30-day placement sweep (`run.py --streams place --tag d30 --rep-delay 30`) | Derived model output |
| `data/figure_inputs/status_quo_540.csv` | Frozen status-quo endpoints | One row per posterior draw aggregated from the ~800 MB scenario `results.jsonl` stream | Derived model output |
| `data/figure_inputs/new_zone_replay_d30_local.csv` | Frozen per-draw new-zone counts | Per-draw scenario replay for the 30-day local-PCR placement panel | Derived model output |
| `data/figure_inputs/posterior_predictive.npz` | Frozen posterior predictive trajectories and observed overlays | Posterior predictive arrays and SitRep-derived observed arrays written by `bdbv.engine` for the appendix model-fit panel | Derived model output plus public INSP/CUOSP SitRep-derived aggregate arrays |
| `data/figure_inputs/observed_at_cutoff.json` | Frozen observed cut-off totals | National observed totals at the 2026-08-02 forecast cut-off | Public INSP/CUOSP SitRep-derived aggregate data |
| `data/figure_inputs/placement_map.geojson` | Frozen placement-map geometry and burdens | Health-zone geometries from `BDBV2026-Data` joined to one illustrative 20-lab, 30-day posterior draw | GRID3 COD Health Zones v8.0 via `BDBV2026-Data`, licensed CC BY 4.0, with fields pruned to derived model outputs and lab flags |
| `data/figure_inputs/placement_map_metadata.json` | Placement-map labels and draw metadata | Metadata for `placement_map.geojson` | Derived model output |
| `bdbv/province_covariates.csv` | Static transformed province covariates | Normalized `s_disruption` score derived from province-level ACLED event rates | ACLED (Armed Conflict Location & Event Data), <https://www.acleddata.com>, accessed 2026-08-05; the CSV retains only transformed province-level scores |
| `bdbv/province_covariates_metadata.json` | Static province-covariate metadata | Machine-readable provenance and transformation note for `s_disruption` | ACLED-derived metadata for a transformed province-level score |
