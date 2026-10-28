"""SYNTHETIC weekly inventory series for the offline demo and tests.

These numbers are generated, not EIA data. They exist so the full pipeline can run in CI
without keys or network access. Levels and volatilities are only loosely realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_series(start: str = "2015-01-02", end: str = "2026-10-16", seed: int = 7) -> pd.DataFrame:
    weeks = pd.date_range(start, end, freq="7D")
    rng = np.random.default_rng(seed)
    n = len(weeks)
    week_of_year = weeks.isocalendar().week.to_numpy(dtype=float)
    phase = 2 * np.pi * week_of_year / 52.0

    def stock(level0, target, season_amp, sd, kappa, shift, floor):
        x = np.empty(n)
        x[0] = level0
        for t in range(1, n):
            drift = season_amp * np.sin(phase[t] + shift) + kappa * (target - x[t - 1])
            x[t] = max(floor, x[t - 1] + drift + rng.normal(0.0, sd))
        return np.round(x)

    frame = pd.DataFrame(
        {
            "crude": stock(440_000, 440_000, 2_500, 4_000, 0.03, 0.0, 300_000),
            "gasoline": stock(235_000, 230_000, 1_800, 2_200, 0.05, 1.5, 180_000),
            "distillate": stock(135_000, 130_000, 1_500, 2_000, 0.05, -1.0, 90_000),
            "cushing": stock(40_000, 35_000, 300, 1_200, 0.05, 0.5, 15_000),
            "refinery_util": np.clip(
                89.0 + 4.0 * np.sin(phase - 1.0) + rng.normal(0.0, 1.8, n), 70.0, 97.0
            ).round(1),
        },
        index=weeks,
    )
    frame.index.name = "week"
    return frame
