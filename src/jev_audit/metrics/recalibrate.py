"""Is the miscalibration fixable? Out-of-time Platt and isotonic recalibration.

Items are ordered by week and split into folds; fold k is recalibrated with a map fitted on
folds 0..k-1 only (expanding window), so no future information leaks into the fix. Fold 0 is
used only for fitting and is excluded from the before/after comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from jev_audit.metrics.binary import brier, ece


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_predict(method: str, p_train, y_train, p_test) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return p_test  # cannot fit a map from one class; leave unchanged
    if method == "platt":
        model = LogisticRegression(C=1e6)
        model.fit(_logit(p_train).reshape(-1, 1), y_train)
        return model.predict_proba(_logit(p_test).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p_train, y_train)
        return model.predict(p_test)
    raise ValueError(f"unknown method {method!r}")


def cross_fit(y, p, order, method: str, n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (recalibrated p, mask of evaluated items) in the original item order."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.argsort(np.asarray(order), kind="mergesort")
    folds = np.array_split(idx, n_folds)
    out = p.copy()
    mask = np.zeros(len(p), dtype=bool)
    for k in range(1, len(folds)):
        train = np.concatenate(folds[:k])
        test = folds[k]
        if len(test) == 0:
            continue
        out[test] = _fit_predict(method, p[train], y[train], p[test])
        mask[test] = True
    return out, mask


def recalibration_report(y, p, order, n_folds: int = 5, n_bins: int = 10) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    platt, mask = cross_fit(y, p, order, "platt", n_folds)
    iso, _ = cross_fit(y, p, order, "isotonic", n_folds)
    if mask.sum() < 10:
        return {"n_eval": int(mask.sum())}
    ym = y[mask]
    return {
        "n_eval": int(mask.sum()),
        "ece_raw": ece(ym, p[mask], n_bins),
        "ece_platt": ece(ym, platt[mask], n_bins),
        "ece_isotonic": ece(ym, iso[mask], n_bins),
        "brier_raw": brier(ym, p[mask]),
        "brier_platt": brier(ym, platt[mask]),
        "brier_isotonic": brier(ym, iso[mask]),
    }
