from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGED = [
    ROOT / "data/BDBV2026-Data/data/insp_sitrep/processed",
    ROOT / "data/BDBV2026-Data/data/osrm/processed",
    ROOT / "data/flowminder_hdx/processed/flowminder_hdx__outflow_2026_03.matrix.csv",
]


@unittest.skipUnless(all(p.exists() for p in STAGED), "outbreak and mobility data are not staged")
class StatusQuoPruneTests(unittest.TestCase):
    """Replaying a recorded tree with no intervention must reproduce it exactly.

    Every averted-burden number is a paired difference against this identity, so a drift
    between `simulate_tree`'s recording and `prune`'s re-evaluation would bias them all.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # The engine resolves its config and output paths from the entry point that
        # imports it; import it as the scenario sweep does.
        argv0 = sys.argv[0]
        sys.argv[0] = str(ROOT / "scenarios" / "run.py")
        try:
            from bdbv import engine as E
            from bdbv.model.tree import prune
            from scenarios import lib
        finally:
            sys.argv[0] = argv0
        cfg = yaml.safe_load((ROOT / "scenarios" / "config.yaml").read_text())
        shared = E.build_context(cfg)["shared"]
        E.init_worker(shared)
        ctx = lib.build_ctx(cfg, shared)
        post = dict(np.load(ROOT / "data/figure_inputs/sq_draws.npz"))
        cls.trees = []
        for i in range(2):
            tree, params, _ = lib.build_tree(
                post, i, shared, ctx["horizon_day"], cutoff_day=ctx["cutoff_day"], k=0
            )
            cls.trees.append((tree, params))
        cls.prune = staticmethod(prune)

    def test_status_quo_prune_reproduces_the_recorded_tree(self) -> None:
        for i, (tree, params) in enumerate(self.trees):
            with self.subTest(draw=i):
                self.assertGreater(tree["n"], 0)
                out = self.prune(tree, params, scenario=None, deploy_day=np.inf)
                self.assertFalse(out["removed"].any())
                np.testing.assert_array_equal(out["safe_burial"], tree["sq_safe_burial"])
                np.testing.assert_array_equal(out["got_pcr"], tree["sq_got_pcr"])
                np.testing.assert_array_equal(out["confirm_abs"], tree["sq_confirm_abs"])
                np.testing.assert_array_equal(out["iso_cutoff"], tree["sq_iso_cutoff"])


if __name__ == "__main__":
    unittest.main()
