"""Statistics for the harness, pure Python.

numpy is not available in this environment and, more usefully, is not needed:
a bootstrap on a few hundred cases is a few hundred thousand list lookups.

The reason this module exists at all is that eval dashboards report point
estimates and humans read a 2 point move as a regression. On a 23 case set, one
flipped case is 4.3 points. The bootstrap interval is what lets the CI gate say
"this move is inside the noise floor of a set this small" and let the merge
through, instead of training the team to ignore the gate.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

BOOTSTRAP_ITERATIONS = 2000
DEFAULT_SEED = 20260910


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95,
                 iterations: int = BOOTSTRAP_ITERATIONS,
                 seed: int = DEFAULT_SEED) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Non-parametric on purpose. Per-case scores are mostly 0/1, so the sampling
    distribution of the mean is binomial and a normal approximation is wrong at
    the small n and extreme p (0.9 and up) that eval sets live at.

    The seed is fixed so the same report produces the same interval twice. A CI
    gate whose verdict changes on re-run is a gate people re-run until it passes.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    pool = list(values)
    means: List[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += pool[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * iterations) - 1)]
    hi = means[min(iterations - 1, int((1.0 - alpha) * iterations))]
    return (lo, hi)


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Correlation coefficient. Used to measure judge verbosity bias.

    Returns 0.0 when either series is constant, which is the honest answer:
    with no variance there is no relationship to detect, not a perfect one.
    """
    if len(xs) != len(ys):
        raise ValueError("series must be the same length")
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def summarise(values: Sequence[float], confidence: float = 0.95,
              seed: int = DEFAULT_SEED) -> Dict[str, float]:
    """Point estimate plus interval, in the shape the report and gate consume."""
    lo, hi = bootstrap_ci(values, confidence=confidence, seed=seed)
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "ci_width": round(hi - lo, 4),
    }
