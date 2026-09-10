"""The statistics behind the promotion gate, in pure Python.

No scipy, and not only because it will not install here. The test is a
two-proportion z-test and it is about twenty lines; importing a numerical stack
to get a normal CDF hides the one part of a promotion decision that a reviewer
should be able to read and disagree with.

Method
------
Two-proportion z-test with a pooled proportion, the standard test for comparing
two conversion rates:

    p_pool = (x_a + x_b) / (n_a + n_b)
    se     = sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z      = (p_b - p_a) / se
    p      = 2 * (1 - Phi(|z|))            (two-sided)

Phi is the standard normal CDF. Two implementations are provided:

  * `normal_cdf` uses `math.erf`, which is in the standard library and is exact
    to double precision: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))). This is what the
    gate calls.
  * `normal_cdf_as26_2_17` is the Abramowitz and Stegun 26.2.17 polynomial
    approximation (Handbook of Mathematical Functions, 1964), absolute error
    below 7.5e-8. It is here because that approximation is what you end up
    writing in an environment without `math.erf`, and because having both lets a
    test assert they agree rather than asserting a number nobody checked.

Two-sided, not one-sided. A one-sided test is tempting because you only want to
promote a winner, and it quietly doubles your false-positive rate in the
direction you are most motivated to believe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function. Exact to double precision."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_cdf_as26_2_17(z: float) -> float:
    """Abramowitz and Stegun 26.2.17. Absolute error below 7.5e-8."""
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    sign = 1.0 if z >= 0 else -1.0
    x = abs(z)
    t = 1.0 / (1.0 + p * x)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    upper_tail = density * poly
    return 0.5 * (1.0 + sign) - sign * upper_tail


@dataclass
class ProportionTest:
    """The full result, not just a boolean, so a refusal can explain itself."""

    successes_a: int
    trials_a: int
    successes_b: int
    trials_b: int
    z: float
    p_value: float

    @property
    def rate_a(self) -> float:
        return self.successes_a / self.trials_a if self.trials_a else 0.0

    @property
    def rate_b(self) -> float:
        return self.successes_b / self.trials_b if self.trials_b else 0.0

    @property
    def absolute_lift(self) -> float:
        return self.rate_b - self.rate_a

    @property
    def relative_lift(self) -> float:
        return (self.rate_b / self.rate_a - 1.0) if self.rate_a else 0.0

    def significant_at(self, alpha: float) -> bool:
        return self.p_value < alpha


def two_proportion_z_test(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int
) -> ProportionTest:
    """Compare two success rates. B is the challenger, A is the control."""
    for value, label in ((trials_a, "trials_a"), (trials_b, "trials_b")):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    for successes, trials, label in (
        (successes_a, trials_a, "a"),
        (successes_b, trials_b, "b"),
    ):
        if not 0 <= successes <= trials:
            raise ValueError(f"successes_{label} must be between 0 and trials_{label}")

    rate_a = successes_a / trials_a
    rate_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / trials_a + 1.0 / trials_b))
    if se == 0.0:
        # Both arms are at exactly 0 or exactly 1. There is no evidence of a
        # difference and no variance to test with; reporting p = 1.0 is the
        # honest answer rather than dividing by zero.
        return ProportionTest(successes_a, trials_a, successes_b, trials_b, z=0.0, p_value=1.0)
    z = (rate_b - rate_a) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return ProportionTest(successes_a, trials_a, successes_b, trials_b, z=z, p_value=p_value)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion.

    Reported alongside the test because a bare success rate hides its own
    uncertainty. The normal approximation interval was rejected: at rates near 0
    or 1, which is where prompt experiments often sit, it produces bounds outside
    [0, 1] and undersells the uncertainty of a small sample.
    """
    if trials <= 0:
        return (0.0, 0.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def required_sample_size(baseline_rate: float, minimum_detectable_effect: float,
                         alpha: float = 0.05, power: float = 0.8) -> Optional[int]:
    """Per-arm sample size for a two-proportion test, normal approximation.

    Used by the demo to say how far off a "not enough data" refusal is, instead
    of just refusing. Returns None when the inputs describe an effect that cannot
    happen, for example a lift that pushes the rate above 1.
    """
    if not 0.0 < baseline_rate < 1.0 or minimum_detectable_effect <= 0:
        return None
    target = baseline_rate + minimum_detectable_effect
    if target >= 1.0:
        return None
    z_alpha = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z_for_two_sided(alpha)
    z_power = 0.8416212335729143 if abs(power - 0.8) < 1e-9 else _z_for_one_sided(power)
    pooled = (baseline_rate + target) / 2.0
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(baseline_rate * (1 - baseline_rate) + target * (1 - target))
    ) ** 2
    return int(math.ceil(numerator / (minimum_detectable_effect ** 2)))


def _z_for_two_sided(alpha: float) -> float:
    return _inverse_normal_cdf(1.0 - alpha / 2.0)


def _z_for_one_sided(power: float) -> float:
    return _inverse_normal_cdf(power)


def _inverse_normal_cdf(p: float, tolerance: float = 1e-10) -> float:
    """Bisection on `normal_cdf`. Slower than a rational approximation and
    short enough to verify by eye, which matters more here than speed."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -10.0, 10.0
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
