"""Climatology baseline: the historical base rate, using only weeks before the state's week.

A forecast that cannot beat climatology on Brier score has no skill. For horizon-1
(next-week) questions this is the bar that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jev_audit.arms.base import Arm, RunContext, dist_answer
from jev_audit.data.states import StateItem


class ClimatologyArm(Arm):
    kind = "climatology"

    def prepare(self, ctx: RunContext) -> None:
        self.alpha = float(self.cfg.extra("alpha", 1.0))
        hw = self.cfg.extra("seasonal_halfwidth")
        self.halfwidth = int(hw) if hw is not None else None
        labels = ctx.labels.copy()
        labels["t"] = pd.to_datetime(labels["week"])
        labels["woy"] = labels["t"].dt.isocalendar().week.astype(int)
        self._hist = {qid: grp.sort_values("t") for qid, grp in labels.groupby("qid")}

    def dist(self, qid: str, week: pd.Timestamp) -> np.ndarray:
        """Smoothed class frequencies from label weeks strictly before `week`.

        For a horizon-1 label dated s (resolved by the report for s + 1), s < week means it
        was already public when the report for `week` came out, so there is no look-ahead.
        """
        q = self.bank[qid]
        k = q.n_classes
        counts = np.zeros(k)
        hist = self._hist.get(qid)
        if hist is not None:
            past = hist[hist["t"] < week]
            if self.halfwidth is not None and len(past):
                target = int(week.isocalendar().week)
                gap = np.abs(past["woy"].to_numpy() - target)
                gap = np.minimum(gap, 52 - gap)
                past = past[gap <= self.halfwidth]
            if len(past):
                counts = np.bincount(past["y"].to_numpy(dtype=int), minlength=k)[:k].astype(float)
        return (counts + self.alpha) / (counts.sum() + k * self.alpha)

    def predict_one(self, state: StateItem):
        week = pd.Timestamp(state.week)
        answers = {q.id: dist_answer(q, self.dist(q.id, week)) for q in self.bank}
        return self.result(state, answers=answers, latency_s=0.0)
