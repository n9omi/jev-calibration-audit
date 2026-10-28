"""Ground-truth labels computed from the EIA series (never from model output)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jev_audit.questions import QuestionBank

LABEL_COLUMNS = ["week", "qid", "horizon", "y", "y_label", "x"]


def regular_weekly(series: pd.DataFrame) -> pd.DataFrame:
    """Reindex to an unbroken 7-day grid so a missing week yields NaN, not a 2-week diff."""
    series = series.sort_index()
    grid = pd.date_range(series.index.min(), series.index.max(), freq="7D")
    off_grid = series.index.difference(grid)
    if len(off_grid):
        raise ValueError(f"Series has dates off the weekly grid, e.g. {off_grid[:3].tolist()}")
    out = series.reindex(grid)
    out.index.name = "week"
    return out


def compute_labels(series: pd.DataFrame, bank: QuestionBank) -> pd.DataFrame:
    """Long table: one row per (report week, question) with the realized answer.

    `x` is the underlying quantity (change in kbbl, or level) for draw/bucket questions,
    kept for diagnostics such as "are errors concentrated in small changes?".
    """
    s = regular_weekly(series)
    rows: list[tuple] = []
    for q in bank:
        lab = q.label
        missing = [name for name in lab.series_list if name not in s.columns]
        if missing:
            continue
        if lab.fn in ("draw", "bucket"):
            x = s[lab.series_list[0]]
            if lab.fn == "draw" or lab.transform == "diff":
                x = x.diff()
            x = x.shift(-lab.lead)
            if lab.fn == "draw":
                y = (x < 0).astype(float)
            else:
                y = pd.Series(np.digitize(x.fillna(0.0), lab.edges), index=x.index, dtype=float)
            y = y.where(x.notna())
            xs = x
        else:  # argmax_abs_pct_change
            pct = pd.DataFrame({k: s[k].diff() / s[k].shift(1) for k in lab.series_list})
            pct = pct.shift(-lab.lead)
            valid = pct.notna().all(axis=1)
            idx = np.argmax(pct.abs().fillna(-1.0).to_numpy(), axis=1)
            y = pd.Series(idx, index=pct.index, dtype=float).where(valid)
            xs = pd.Series(np.nan, index=pct.index)
        for week, val in y.dropna().items():
            k = int(val)
            rows.append((week.date().isoformat(), q.id, q.horizon, k, q.classes[k], float(xs.loc[week])))
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)
