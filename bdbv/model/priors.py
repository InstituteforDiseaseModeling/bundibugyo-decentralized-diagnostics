"""Prior-sampling helpers: draw from, and evaluate, the prior specs in config.yaml."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import lognorm as _lognorm
from scipy.stats import norm as _norm
from scipy.stats import truncnorm as _truncnorm


def sample_prior(spec: dict, rng: np.random.Generator) -> float:
    """Sample one value from a prior spec dict (see experiment config.yaml)."""
    if "fixed" in spec:
        return float(spec["fixed"])
    d = spec["dist"]
    if d == "normal":
        mean = float(spec["mean"])
        sd = float(spec["sd"])
        lo = float(spec.get("min", -np.inf))
        hi = float(spec.get("max", np.inf))
        return float(
            _truncnorm.rvs(
                (lo - mean) / sd,
                (hi - mean) / sd,
                loc=mean,
                scale=sd,
                random_state=rng,
            )
        )
    if d == "lognormal":
        return float(rng.lognormal(spec["meanlog"], spec["sdlog"]))
    if d == "beta":
        return float(rng.beta(spec["a"], spec["b"]))
    if d == "uniform":
        return float(rng.uniform(spec["min"], spec["max"]))
    if d == "loguniform":
        # Scale-free prior, for half-saturation constants spanning decades.
        return float(np.exp(rng.uniform(np.log(spec["min"]), np.log(spec["max"]))))
    raise ValueError(f"unknown dist {d!r}")


def log_prior(spec: dict, x: float) -> float:
    """Log-density of a scalar prior spec at x. Returns -inf outside support.
    Used by the MCMC sampler."""
    if "fixed" in spec:
        return 0.0 if x == float(spec["fixed"]) else -np.inf
    d = spec["dist"]
    if d == "normal":
        lo = spec.get("min", -np.inf)
        hi = spec.get("max", np.inf)
        if x < lo or x > hi:
            return -np.inf
        return float(_norm.logpdf(x, loc=spec["mean"], scale=spec["sd"]))
    if d == "lognormal":
        if x <= 0:
            return -np.inf
        return float(_lognorm.logpdf(x, s=spec["sdlog"], scale=np.exp(spec["meanlog"])))
    if d == "beta":
        if x <= 0 or x >= 1:
            return -np.inf
        return float(_beta.logpdf(x, spec["a"], spec["b"]))
    if d == "uniform":
        lo, hi = float(spec["min"]), float(spec["max"])
        if x < lo or x > hi:
            return -np.inf
        return -np.log(hi - lo)
    if d == "loguniform":
        lo, hi = float(spec["min"]), float(spec["max"])
        if x < lo or x > hi:
            return -np.inf
        return -np.log(x) - np.log(np.log(hi) - np.log(lo))
    raise ValueError(f"unknown dist {d!r}")


def gamma_shape_scale(mean: float, sd: float) -> tuple[float, float]:
    """Gamma (shape, scale) matching a target mean and sd."""
    var = sd**2
    return mean**2 / var, var / mean


def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """LogNormal (meanlog, sdlog) matching a target mean and sd."""
    sdlog = np.sqrt(np.log(1 + (sd / mean) ** 2))
    return np.log(mean) - 0.5 * sdlog**2, sdlog
