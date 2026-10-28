"""Rules baseline: parse the numbers out of the text and answer deterministically.

Where parsing fails, and for every horizon-1 question, it falls back to climatology.
On templated horizon-0 questions it should be (near) perfect; that is the point — it shows
which questions are pure extraction, so model differences there are about reading, not skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jev_audit.arms.base import Arm, RunContext, dist_answer
from jev_audit.arms.climatology import ClimatologyArm
from jev_audit.data.parse import parse_report
from jev_audit.data.states import StateItem
from jev_audit.questions import QuestionSpec


def deterministic_answer(q: QuestionSpec, parsed: dict[str, dict[str, float]]) -> int | None:
    lab = q.label
    if lab.lead != 0:
        return None
    if lab.fn == "draw":
        change = parsed.get(lab.series_list[0], {}).get("change")
        return None if change is None else int(change < 0)
    if lab.fn == "bucket":
        found = parsed.get(lab.series_list[0], {})
        x = found.get("change") if lab.transform == "diff" else found.get("level")
        return None if x is None else int(np.digitize([x], lab.edges)[0])
    pcts = []
    for name in lab.series_list:
        found = parsed.get(name, {})
        if "change" not in found or not found.get("prev"):
            return None
        pcts.append(abs(found["change"] / found["prev"]))
    return int(np.argmax(pcts))


class RulesArm(Arm):
    kind = "rules"

    def __init__(self, name, cfg, bank):
        super().__init__(name, cfg, bank)
        self.eps = float(cfg.extra("eps", 0.01))
        self._fallback = ClimatologyArm(f"{name}_fallback", cfg, bank)

    def prepare(self, ctx: RunContext) -> None:
        self._fallback.prepare(ctx)

    def _onehot(self, k: int, n: int) -> np.ndarray:
        dist = np.full(n, self.eps / (n - 1))
        dist[k] = 1.0 - self.eps
        return dist

    def predict_one(self, state: StateItem):
        parsed = parse_report(state.text)
        week = pd.Timestamp(state.week)
        answers = {}
        for q in self.bank:
            k = deterministic_answer(q, parsed)
            if k is None:
                answers[q.id] = dist_answer(q, self._fallback.dist(q.id, week), abstained=True)
            else:
                answers[q.id] = dist_answer(q, self._onehot(k, q.n_classes), abstained=False)
        return self.result(state, answers=answers, latency_s=0.0)
