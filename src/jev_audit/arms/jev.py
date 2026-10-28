"""Jev arm: TypeSafe's System One model through langchain-typesafe's TypeSafeClassifier.

All questions about a state go out in ONE request; Jev evaluates them in parallel. Answers
come back as probabilities: Noul -> P(yes); Choice -> per-label probabilities plus a
`confidence` (a concentration measure, NOT the chosen label's probability); Score -> per-level
probabilities, an expected `score`, and `confidence`. We score the probabilities and keep the
confidence separately so we can test whether it is a useful abstention signal.
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Any

from jev_audit.arms.base import Arm, dist_answer, noul_answer
from jev_audit.data.states import StateItem


class JevArm(Arm):
    kind = "jev"

    def __init__(self, name, cfg, bank, client: Any = None):
        super().__init__(name, cfg, bank)
        from langchain_typesafe import TypeSafeClassifier

        kwargs: dict[str, Any] = {"model": cfg.model or "jev-latest"}
        if client is not None:  # e.g. an httpx2.Client with a MockTransport in tests
            kwargs["client"] = client
        if cfg.extra("timeout") is not None:
            kwargs["timeout"] = float(cfg.extra("timeout"))
        with warnings.catch_warnings():
            # The integration is marked beta (documented in the README); keep run logs clean.
            warnings.filterwarnings("ignore", message=".*TypeSafeClassifier.*beta.*")
            self.classifier = TypeSafeClassifier(**kwargs)
        self.questions = bank.to_typesafe()
        self.max_retries = int(cfg.extra("max_retries", 5))

    def _invoke(self, request: dict, config: dict):
        from langchain_typesafe.client import (
            TypeSafeAPIConnectionError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return self.classifier.invoke(request, config=config)
            except TypeSafeRateLimitError as exc:
                if attempt == self.max_retries:
                    raise
                wait = (exc.retry_after_ms / 1000.0) if exc.retry_after_ms is not None else delay
            except (TypeSafeInternalServerError, TypeSafeAPIConnectionError):
                if attempt == self.max_retries:
                    raise
                wait = delay
            time.sleep(min(wait, 60.0) * (1.0 + 0.1 * random.random()))
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    def predict_one(self, state: StateItem):
        config = {
            "run_name": f"audit:{self.name}",
            "tags": ["jev-calibration-audit", self.name, state.view],
            "metadata": {"state_id": state.state_id, "view": state.view, "arm": self.name},
        }
        start = time.perf_counter()
        resp = self._invoke({"state": state.text, "questions": self.questions}, config)
        latency = time.perf_counter() - start

        answers: dict[str, dict] = {}
        missing = []
        for q in self.bank:
            ans = resp.answers.get(q.id)
            if ans is None:
                missing.append(q.id)
                continue
            if q.type == "noul":
                answers[q.id] = noul_answer(ans.noul)
            elif q.type == "choice":
                probs = [ans.probabilities.get(c, 0.0) for c in q.classes]
                answers[q.id] = dist_answer(q, probs, confidence=ans.confidence, chosen=ans.choice)
            else:
                probs = [ans.probabilities.get(i, 0.0) for i in range(q.n_classes)]
                answers[q.id] = dist_answer(q, probs, confidence=ans.confidence, score=ans.score)
        return self.result(
            state,
            model=resp.model,
            answers=answers,
            latency_s=latency,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            error=f"missing answers: {missing}" if missing else None,
        )
