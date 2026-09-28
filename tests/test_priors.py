from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import kstest, truncnorm

from bdbv.model.priors import log_prior, sample_prior


class PriorTests(unittest.TestCase):
    def test_bounded_normal_sampling_matches_log_prior_support(self) -> None:
        spec = {"dist": "normal", "mean": 0.0, "sd": 1.0, "min": -2.0, "max": 2.0}
        rng = np.random.default_rng(42)

        draws = np.array([sample_prior(spec, rng) for _ in range(2_000)])
        self.assertTrue(np.all([np.isfinite(log_prior(spec, x)) for x in draws]))
        self.assertTrue(np.all(draws > spec["min"]))
        self.assertTrue(np.all(draws < spec["max"]))

        a = (spec["min"] - spec["mean"]) / spec["sd"]
        b = (spec["max"] - spec["mean"]) / spec["sd"]
        result = kstest(
            draws,
            truncnorm(a, b, loc=spec["mean"], scale=spec["sd"]).cdf,
        )
        self.assertGreater(result.pvalue, 0.01)


if __name__ == "__main__":
    unittest.main()
