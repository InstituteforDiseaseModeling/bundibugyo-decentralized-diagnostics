from __future__ import annotations

import unittest

import numpy as np

from scenarios.draw_split import split_greedy_and_score_draws


class ScenarioTests(unittest.TestCase):
    def test_greedy_prepass_and_scoring_draws_are_disjoint(self) -> None:
        post = {
            "R0": np.arange(6),
            "sim_seed": np.arange(10, 16),
        }

        train, score, n_pre, n_score, train_rows, score_rows = split_greedy_and_score_draws(
            post,
            requested_score=4,
            requested_pre=2,
            seed=42,
        )

        self.assertEqual(2, n_pre)
        self.assertEqual(4, n_score)
        self.assertEqual(set(range(6)), set(train_rows) | set(score_rows))
        self.assertEqual(set(), set(train_rows) & set(score_rows))
        np.testing.assert_array_equal(train_rows, train["R0"])
        np.testing.assert_array_equal(score_rows, score["R0"])

    def test_full_run_refuses_overlapping_subsample(self) -> None:
        post = {
            "R0": np.arange(5),
            "sim_seed": np.arange(10, 15),
        }

        with self.assertRaises(ValueError) as ctx:
            split_greedy_and_score_draws(
                post,
                requested_score=4,
                requested_pre=2,
                seed=42,
            )
        self.assertIn("extract_subsample.py with --n 6", str(ctx.exception))

    def test_one_draw_smoke_test_keeps_one_heldout_draw(self) -> None:
        post = {
            "R0": np.arange(3),
            "sim_seed": np.arange(10, 13),
        }

        train, score, n_pre, n_score, train_rows, score_rows = split_greedy_and_score_draws(
            post,
            requested_score=1,
            requested_pre=500,
            seed=42,
        )

        self.assertEqual(2, n_pre)
        self.assertEqual(1, n_score)
        self.assertEqual(2, train_rows.size)
        self.assertEqual(1, score_rows.size)
        self.assertEqual(set(range(3)), set(train_rows) | set(score_rows))
        np.testing.assert_array_equal(train_rows, train["R0"])
        np.testing.assert_array_equal(score_rows, score["R0"])

    def test_split_preserves_static_arrays(self) -> None:
        zones = np.array(["A", "B", "C"])
        post = {
            "R0": np.arange(6),
            "sim_seed": np.arange(10, 16),
            "zones": zones,
        }

        train, score, *_ = split_greedy_and_score_draws(
            post,
            requested_score=4,
            requested_pre=2,
            seed=42,
        )

        np.testing.assert_array_equal(zones, train["zones"])
        np.testing.assert_array_equal(zones, score["zones"])


if __name__ == "__main__":
    unittest.main()
