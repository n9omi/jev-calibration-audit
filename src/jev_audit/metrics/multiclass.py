"""Metrics for Choice (nominal) and Score (ordinal) probability vectors.

y: integer class indices, shape (n,). P: probabilities, shape (n, K), rows summing to 1.
"""

from __future__ import annotations

import numpy as np

from jev_audit.metrics.binary import auroc, ece


def _arrays(y, P) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != y.shape[0]:
        raise ValueError("P must be (n, K) with n == len(y)")
    return y, P


def onehot(y, k: int) -> np.ndarray:
    return np.eye(k)[np.asarray(y, dtype=int)]


def multiclass_brier_per(y, P) -> np.ndarray:
    """Sum over classes of squared error; ranges 0 (perfect) to 2."""
    y, P = _arrays(y, P)
    return np.sum((P - onehot(y, P.shape[1])) ** 2, axis=1)


def nll(y, P, eps: float = 1e-6) -> float:
    y, P = _arrays(y, P)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))


def rps_per(y, P) -> np.ndarray:
    """Ranked probability score, normalized to [0, 1]. The proper score for ordered levels:
    putting mass on a level next to the truth costs less than putting it far away."""
    y, P = _arrays(y, P)
    k = P.shape[1]
    cdf_p = np.cumsum(P, axis=1)[:, :-1]
    cdf_y = np.cumsum(onehot(y, k), axis=1)[:, :-1]
    return np.sum((cdf_p - cdf_y) ** 2, axis=1) / (k - 1)


def top_label_ece(y, P, n_bins: int = 10) -> float:
    y, P = _arrays(y, P)
    return ece((P.argmax(axis=1) == y).astype(float), P.max(axis=1), n_bins)


def classwise_ece(y, P, n_bins: int = 10) -> float:
    y, P = _arrays(y, P)
    return float(np.mean([ece((y == k).astype(float), P[:, k], n_bins) for k in range(P.shape[1])]))


def multiclass_summary(y, P, ordinal: bool, n_bins: int = 10, confidence=None) -> dict[str, float]:
    y, P = _arrays(y, P)
    correct = (P.argmax(axis=1) == y).astype(float)
    out = {
        "n": int(len(y)),
        "accuracy": float(correct.mean()) if len(y) else float("nan"),
        "mbrier": float(np.mean(multiclass_brier_per(y, P))),
        "nll": nll(y, P),
        "top_ece": top_label_ece(y, P, n_bins),
        "classwise_ece": classwise_ece(y, P, n_bins),
        # Is the top probability a useful signal for "this answer is right"?
        "maxprob_auroc": auroc(correct, P.max(axis=1)),
    }
    if ordinal:
        levels = np.arange(P.shape[1])
        out["rps"] = float(np.mean(rps_per(y, P)))
        out["level_mae"] = float(np.mean(np.abs(P @ levels - y)))
    if confidence is not None:
        conf = np.asarray(confidence, dtype=float)
        if np.isfinite(conf).all():
            out["confidence_auroc"] = auroc(correct, conf)
    return out
