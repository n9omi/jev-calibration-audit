"""Common interface for every classifier arm and the normalized answer format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from jev_audit.config import ArmConfig
from jev_audit.data.states import StateItem
from jev_audit.questions import QuestionBank, QuestionSpec


@dataclass
class RunContext:
    """What an arm may see before predicting. Only `climatology`, `rules` (for its fallback)
    and `mock` arms use the label history, and climatology only uses weeks strictly earlier
    than the state being predicted."""

    labels: pd.DataFrame


@dataclass
class ArmResult:
    state_id: str
    week: str
    view: str
    arm: str
    model: str | None
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ normalized answers
# noul   -> {"type": "noul",   "p": P(true)}
# choice -> {"type": "choice", "probs": [p_k in question.classes order], "confidence": ...}
# score  -> {"type": "score",  "probs": [p_level_0, ...], "confidence": ..., "score": ...}


def normalize(probs: Sequence[float]) -> tuple[list[float], float]:
    """Clip negatives and rescale to sum to 1. Returns (probs, raw_sum) for diagnostics."""
    arr = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        return [1.0 / len(arr)] * len(arr), total
    return (arr / total).tolist(), total


def noul_answer(p: float, **extra: Any) -> dict[str, Any]:
    raw = float(p)
    return {"type": "noul", "p": float(np.clip(raw, 0.0, 1.0)), **extra}


def dist_answer(q: QuestionSpec, probs: Sequence[float], **extra: Any) -> dict[str, Any]:
    """Answer from a full distribution over `q.classes` (index 1 = true for noul)."""
    if q.type == "noul":
        return noul_answer(probs[1], **extra)
    normed, raw_sum = normalize(probs)
    return {"type": q.type, "probs": normed, "raw_sum": raw_sum, **extra}


class Arm(ABC):
    kind: str = "base"
    requires_labels: bool = False

    def __init__(self, name: str, cfg: ArmConfig, bank: QuestionBank):
        self.name = name
        self.cfg = cfg
        self.bank = bank

    @property
    def model_name(self) -> str:
        return self.cfg.model or self.kind

    def prepare(self, ctx: RunContext) -> None:  # noqa: B027 - optional hook
        """Called once before predicting (e.g. to load label history)."""

    @abstractmethod
    def predict_one(self, state: StateItem) -> ArmResult: ...

    def result(self, state: StateItem, **kwargs: Any) -> ArmResult:
        return ArmResult(
            state_id=state.state_id,
            week=state.week,
            view=state.view,
            arm=self.name,
            model=kwargs.pop("model", self.model_name),
            **kwargs,
        )

    def _safe(self, state: StateItem) -> ArmResult:
        try:
            return self.predict_one(state)
        except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
            return self.result(state, error=f"{type(exc).__name__}: {exc}")

    def predict_many(
        self, states: Iterable[StateItem], max_concurrency: int | None = None
    ) -> Iterator[ArmResult]:
        workers = max(1, max_concurrency or self.cfg.max_concurrency)
        states = list(states)
        if workers == 1:
            for state in states:
                yield self._safe(state)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._safe, s) for s in states]
            for fut in as_completed(futures):
                yield fut.result()
