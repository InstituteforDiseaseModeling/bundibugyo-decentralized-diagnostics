"""Negative-Binomial composite pseudo-likelihood for the BDBV fit.

The observation layer emits EXPECTED counts (ascertainment / IFR applied as
expected fractions, deterministic given a latent trajectory); observed counts
are scored against those expectations with a Negative-Binomial (Gamma-Poisson)
kernel — overdispersed, Funk-consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

# Floor on expected counts to avoid -inf logpmf when an empty simulated cell
# meets a zero observation. Small enough that any non-trivial expected count
# dominates.
_EXPECTED_FLOOR = 1e-6


def nb_loglik(obs, expected, dispersion: float) -> float:
    """Summed NB log-likelihood. NB has mean=expected, size=dispersion
    (smaller dispersion ⇒ more overdispersion). Expected is floored at
    ``_EXPECTED_FLOOR`` so empty simulated cells don't return -inf for zero obs.
    """
    mu = np.maximum(np.asarray(expected, float), _EXPECTED_FLOOR)
    r = float(dispersion)
    p = r / (r + mu)
    return float(nbinom.logpmf(np.asarray(obs, float), r, p).sum())


def normal_loglik(obs, expected, sd: float) -> float:
    """Summed Normal log-likelihood, for continuous derived summaries with a published
    uncertainty — the effective-Rt anchor, where the target is an epiforecasts credible
    band rather than a count."""
    sd = max(float(sd), 1e-9)
    z = (np.asarray(obs, float) - np.asarray(expected, float)) / sd
    return float((-0.5 * z**2 - np.log(sd) - 0.5 * np.log(2 * np.pi)).sum())


def nb_lower_bound_loglik(obs, expected, dispersion: float) -> float:
    """One-sided (soft lower-bound) NB log-likelihood.

    The observed count is treated as a LOWER BOUND on the truth: the terminal
    death bin is under-counted via forward-confirmation incompleteness —
    community deaths not yet swabbed sit in no cell yet. We therefore penalise
    the model ONLY where it UNDER-predicts (expected < obs); where the model
    meets or exceeds the observed floor (expected >= obs) the cell contributes
    0, imposing no penalty for over-shooting a target we know is incomplete.

    Same NB(mean=expected, size=dispersion) kernel as ``nb_loglik`` for the
    penalised branch, so it composes at the same dispersion. Expected is floored
    at ``_EXPECTED_FLOOR`` to avoid -inf.
    """
    obs = np.asarray(obs, float)
    mu = np.maximum(np.asarray(expected, float), _EXPECTED_FLOOR)
    r = float(dispersion)
    p = r / (r + mu)
    ll = nbinom.logpmf(obs, r, p)
    # Only the under-prediction branch contributes; over-prediction is free.
    return float(np.where(mu < obs, ll, 0.0).sum())


def beta_binomial_loglik(k, n, p_model, s: float, min_p: float = 1e-4) -> float:
    """Beta-Binomial log-likelihood for observed count ``k`` out of ``n``, given
    the model's predicted success PROBABILITY ``p_model`` and concentration
    ``s`` (an effective prior sample size). Parameterised a = s*p, b = s*(1-p):
    as ``s`` -> inf this approaches Binomial(n, p); small ``s`` tolerates
    overdispersion.

    This scores the CORE-vs-PERIPHERY spatial split — ``k`` = observed
    'other' cases, ``n`` = observed scored total, ``p_model`` = the model's
    'other' share. The forecast-critical quantity is exactly this share (how
    fast transmission jumps out of the outbreak core into the rest of DRC), so
    it gets its own likelihood term rather than being one cell among 16 in the
    DM. ``s`` ALONE sets how hard the split is enforced — there is deliberately
    no separate component weight, so the tolerance is not double-counted. At
    s=300, a model 'other' share of 2x the observed fraction costs ~6.3
    log-units; a 33% overshoot costs ~0.7 (free wiggle).
    """
    from scipy.stats import betabinom

    k = round(float(k))
    n = round(float(n))
    if n <= 0 or k < 0 or k > n:
        return 0.0
    p = float(np.clip(p_model, min_p, 1.0 - min_p))
    a = s * p
    b = s * (1.0 - p)
    return float(betabinom.logpmf(k, n, a, b))


def composite_loglik(components, dispersion: float, weights=None) -> float:
    """Sum of weighted per-component NB log-likelihoods.

    ``components`` is a list of (obs_array, expected_array) pairs.
    """
    if weights is None:
        weights = [1.0] * len(components)
    return sum(w * nb_loglik(o, e, dispersion) for w, (o, e) in zip(weights, components))


def dirichlet_multinomial_loglik(
    obs_counts, model_mean, alpha: float, min_share: float = 1e-4
) -> float:
    """Dirichlet-multinomial log-likelihood for the observed COUNT VECTOR
    ``obs_counts`` given model expected shares derived from ``model_mean`` and
    concentration ``alpha``.

    Model shares ``p_i = model_mean_i / sum(model_mean)``, clipped at
    ``min_share`` then renormalised (avoids ``log(0)`` when the model produces
    zero mass in a zone that has observed cases).

    Formula (from the DM pmf, dropping the ``multinomial(N, obs)`` combinatorial
    term that doesn't depend on model parameters):

        logL = loggamma(alpha) - loggamma(N + alpha)
             + sum_i [ loggamma(alpha * p_i + obs_i) - loggamma(alpha * p_i) ]

    Higher ``alpha`` = tighter around model shares (approaches multinomial as
    alpha -> inf). Lower ``alpha`` = more overdispersion tolerance.
    """
    from scipy.special import gammaln

    obs_counts = np.asarray(obs_counts, dtype=float)
    N = float(obs_counts.sum())
    if N <= 0:
        return 0.0
    model_mean = np.asarray(model_mean, dtype=float)
    total = model_mean.sum()
    if total <= 0:
        return -1e10  # penalise: model produces no mass anywhere
    p = np.maximum(model_mean / total, min_share)
    p = p / p.sum()  # renormalise after clipping
    ap = alpha * p
    ll = gammaln(alpha) - gammaln(N + alpha) + np.sum(gammaln(ap + obs_counts) - gammaln(ap))
    return float(ll)
