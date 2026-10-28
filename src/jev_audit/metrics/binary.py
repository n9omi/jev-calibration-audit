"""Calibration and accuracy metrics for binary (Noul) probabilities.

y: array of 0/1 outcomes. p: array of forecast probabilities P(y = 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    return y, p


def brier_per(y, p) -> np.ndarray:
    y, p = _arrays(y, p)
    return (p - y) ** 2


def brier(y, p) -> float:
    return float(np.mean(brier_per(y, p)))


def log_loss(y, p, eps: float = 1e-6) -> float:
    y, p = _arrays(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y, p) -> float:
    y, p = _arrays(y, p)
    return float(np.mean((p >= 0.5) == (y == 1)))


def _edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"unknown binning strategy {strategy!r}")


def reliability_table(y, p, n_bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """Non-empty bins with forecast mean and observed frequency (the reliability diagram)."""
    y, p = _arrays(y, p)
    edges = _edges(p, n_bins, strategy)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n:
            rows.append((b, edges[b], edges[b + 1], n, float(p[mask].mean()), float(y[mask].mean())))
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "mean_p", "freq"])


def ece(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: count-weighted mean |forecast - observed| over bins."""
    t = reliability_table(y, p, n_bins, strategy)
    if t.empty:
        return float("nan")
    return float(np.sum(t["n"] * np.abs(t["mean_p"] - t["freq"])) / t["n"].sum())


def mce(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    t = reliability_table(y, p, n_bins, strategy)
    return float(np.max(np.abs(t["mean_p"] - t["freq"]))) if len(t) else float("nan")


def murphy_decomposition(y, p, n_bins: int = 10) -> dict[str, float]:
    """Brier = reliability - resolution + uncertainty (+ residual from binning).

    reliability: miscalibration (lower is better); resolution: how much forecasts separate
    outcomes (higher is better); uncertainty: base-rate variance (not the forecaster's doing).
    The residual is exactly zero when every bin holds a single forecast value.
    """
    y, p = _arrays(y, p)
    t = reliability_table(y, p, n_bins, "uniform")
    n = len(y)
    ybar = float(y.mean())
    rel = float(np.sum(t["n"] * (t["mean_p"] - t["freq"]) ** 2) / n)
    res = float(np.sum(t["n"] * (t["freq"] - ybar) ** 2) / n)
    unc = ybar * (1 - ybar)
    return {"rel": rel, "res": res, "unc": unc, "resid": brier(y, p) - (rel - res + unc)}


def spiegelhalter(y, p) -> tuple[float, float]:
    """Spiegelhalter's Z test. H0: the forecasts are calibrated. Returns (z, two-sided p)."""
    y, p = _arrays(y, p)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    if den == 0:
        return float("nan"), float("nan")
    z = float(num / den)
    return z, float(2 * norm.sf(abs(z)))


def auroc(y, p) -> float:
    """Discrimination: probability a random positive gets a higher forecast than a negative."""
    y, p = _arrays(y, p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_summary(y, p, n_bins: int = 10) -> dict[str, float]:
    y, p = _arrays(y, p)
    z, zp = spiegelhalter(y, p)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "accuracy": accuracy(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": ece(y, p, n_bins, "uniform"),
        "ece_quantile": ece(y, p, n_bins, "quantile"),
        "mce": mce(y, p, n_bins, "uniform"),
        "auroc": auroc(y, p),
        "spiegelhalter_z": z,
        "spiegelhalter_p": zp,
        "mean_p": float(p.mean()) if len(p) else float("nan"),
        "sharpness": float(np.mean(np.abs(p - 0.5)) * 2) if len(p) else float("nan"),
    }
    out.update(murphy_decomposition(y, p, n_bins))
    return out
