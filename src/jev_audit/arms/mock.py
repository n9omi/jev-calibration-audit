"""Simulated forecaster for the offline demo and tests.

WARNING: this arm reads the true labels. It exists only to show that the metrics separate a
calibrated forecaster from an overconfident one. Never include it in a real run.

Construction (binary): z = s * (2y - 1) + N(0, 1). The Bayes posterior is
logit P(y=1 | z) = 2 s z + logit(prior), so p = sigmoid(sharpen * that) is calibrated at
sharpen = 1 and overconfident for sharpen > 1. Multiclass: z = s * onehot(y) + N(0, I),
posterior = softmax(s z + log prior).
"""

from __future__ import annotations

import zlib

import numpy as np

from jev_audit.arms.base import Arm, RunContext, dist_answer
from jev_audit.data.states import StateItem


def _h(text: str) -> int:
    return zlib.crc32(text.encode())


class MockArm(Arm):
    kind = "mock"
    requires_labels = True

    def prepare(self, ctx: RunContext) -> None:
        self.skill = float(self.cfg.extra("skill", 1.5))
        self.forecast_skill = float(self.cfg.extra("forecast_skill", 0.25))
        self.sharpen = float(self.cfg.extra("sharpen", 1.0))
        self.seed = int(self.cfg.extra("seed", 0))
        labels = ctx.labels
        self._y = {(w, q): int(y) for w, q, y in zip(labels["week"], labels["qid"], labels["y"], strict=True)}
        self._prior = {}
        for q in self.bank:
            ys = labels.loc[labels["qid"] == q.id, "y"].to_numpy(dtype=int)
            counts = np.bincount(ys, minlength=q.n_classes)[: q.n_classes] + 1.0
            self._prior[q.id] = counts / counts.sum()

    def predict_one(self, state: StateItem):
        answers = {}
        for q in self.bank:
            k = q.n_classes
            y = self._y.get((state.week, q.id))
            if y is None:
                answers[q.id] = dist_answer(q, self._prior[q.id])
                continue
            rng = np.random.default_rng([self.seed, _h(state.state_id), _h(q.id)])
            s = self.skill if q.horizon == 0 else self.forecast_skill
            log_prior = np.log(self._prior[q.id])
            if k == 2:
                z = s * (2 * y - 1) + rng.normal()
                logit = self.sharpen * (2 * s * z + log_prior[1] - log_prior[0])
                p = 1.0 / (1.0 + np.exp(-logit))
                dist = np.array([1.0 - p, p])
            else:
                z = s * np.eye(k)[y] + rng.normal(size=k)
                logits = self.sharpen * (s * z + log_prior)
                dist = np.exp(logits - logits.max())
                dist /= dist.sum()
            answers[q.id] = dist_answer(q, dist)
        return self.result(state, answers=answers, latency_s=0.0)
