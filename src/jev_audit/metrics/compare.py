"""Paired comparison of two forecasters' losses on the same weekly items.

Weekly losses are serially correlated (inventory regimes persist), so we use a
Newey-West (HAC) variance in the Diebold-Mariano test and a circular moving-block bootstrap
for confidence intervals rather than iid formulas.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t


def newey_west_lag(n: int) -> int:
    return max(0, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_var(d, lag: int | None = None) -> float:
    """Long-run variance of a series (Bartlett kernel)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    lag = newey_west_lag(n) if lag is None else lag
    e = d - d.mean()
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float(e[k:] @ e[:-k]) / n
    return max(s, 0.0)


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> tuple[float, float]:
    """DM test of equal expected loss, with the Harvey-Leybourne-Newbold small-sample
    correction (h = 1) and Student-t reference. Negative statistic: A has lower loss."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    var = newey_west_var(d, lag)
    if not var > 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(var / n)
    stat *= math.sqrt((n - 1) / n)  # HLN correction for one-step-ahead
    p = 2.0 * student_t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def block_bootstrap_ci(
    d, n_boot: int = 2000, block: int | None = None, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Circular moving-block bootstrap CI for the mean of `d`."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    block = block or max(1, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare_losses(loss_a, loss_b, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    lo, hi = block_bootstrap_ci(d, n_boot=n_boot, seed=seed)
    stat, p = diebold_mariano(a, b)
    return {
        "n": int(len(d)),
        "loss_a": float(a.mean()),
        "loss_b": float(b.mean()),
        "diff": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "dm_stat": stat,
        "dm_p": p,
    }
