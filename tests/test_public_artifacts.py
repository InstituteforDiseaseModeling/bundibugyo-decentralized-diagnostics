from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP_KEYS = {
    "nom",
    "in_model",
    "sq",
    "random",
    "celf",
    "existing_lab",
    "random_lab",
    "celf_lab",
}


class PublicArtifactTests(unittest.TestCase):
    def test_placement_map_is_pruned_to_figure_fields(self) -> None:
        with (ROOT / "data/figure_inputs/placement_map.geojson").open() as handle:
            geojson = json.load(handle)

        feature_keys = {key for feature in geojson["features"] for key in feature["properties"]}
        self.assertEqual(MAP_KEYS, feature_keys)
        self.assertEqual(519, len(geojson["features"]))

    def test_npz_figure_inputs_load_without_pickle(self) -> None:
        for path in [
            ROOT / "data/figure_inputs/sq_draws.npz",
            ROOT / "data/figure_inputs/posterior_predictive.npz",
            ROOT / "data/inputs/source_checkpoint_iter44000.npz",
        ]:
            with self.subTest(path=path), np.load(path) as artifact:
                self.assertTrue(artifact.files)

        with np.load(ROOT / "data/figure_inputs/sq_draws.npz") as artifact:
            self.assertEqual("U", artifact["zones"].dtype.kind)

    def test_bundled_scenario_checkpoint_names_its_iteration(self) -> None:
        with np.load(ROOT / "data/inputs/source_checkpoint_iter44000.npz") as checkpoint:
            self.assertEqual(44000, int(checkpoint["_iter"][0]))

    def test_province_covariates_only_publish_transformed_acled_score(self) -> None:
        with (ROOT / "bdbv/province_covariates.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(
            ["province", "s_disruption"],
            reader.fieldnames,
        )
        self.assertEqual(26, len(rows))

    def test_hdx_checksums_match_when_hdx_is_staged(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import stage_data

        raw = ROOT / "data/flowminder_hdx/raw" / stage_data.HDX_RELOCATIONS_FILE
        matrix = (
            ROOT / "data/flowminder_hdx/processed" / "flowminder_hdx__outflow_2026_03.matrix.csv"
        )
        if not raw.exists() or not matrix.exists():
            self.skipTest("HDX data are not staged")

        self.assertEqual(stage_data.HDX_RELOCATIONS_SHA256, stage_data.sha256(raw))
        self.assertEqual(stage_data.HDX_MATRIX_SHA256, stage_data.sha256(matrix))

    def test_prepare_scenario_checkpoint_preserves_iteration_name(self) -> None:
        script = ROOT / "scripts/prepare_scenario_checkpoint.py"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "checkpoint_ppc.npz"
            good_dest = tmp_path / "source_checkpoint_iter7.npz"
            bad_dest = tmp_path / "source_checkpoint_iter44000.npz"
            np.savez_compressed(source, _iter=np.array([7]))

            ok = subprocess.run(
                [sys.executable, str(script), "--source", str(source), "--dest", str(good_dest)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertTrue(good_dest.exists())

            bad = subprocess.run(
                [sys.executable, str(script), "--source", str(source), "--dest", str(bad_dest)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("contains _iter=7", bad.stderr)


if __name__ == "__main__":
    unittest.main()
