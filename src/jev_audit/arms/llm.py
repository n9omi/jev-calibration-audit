"""LLM-as-classifier arm: any LangChain chat model asked for probabilities via structured output.

This is the comparison Jev is pitched against: using a general LLM to make the same typed
decisions. Probabilities are *verbalized* (the model writes them down). That is how most teams
use LLM classifiers today, and verbalized probabilities are known to be overconfident — which
is exactly the claim this audit tests. One call per state, all questions together, same
wording as the Jev arm.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, create_model

from jev_audit.arms.base import Arm, dist_answer, noul_answer
from jev_audit.data.states import StateItem
from jev_audit.questions import QuestionBank

SYSTEM_PROMPT = """\
You are an energy-market analyst acting as a probabilistic classifier. You will read a weekly \
U.S. petroleum inventory summary and answer several questions about it.

Report calibrated probabilities: among all the statements you give 0.8, about 80% should turn \
out true. Use 0 or 1 only when the report text makes the answer certain. When the text does \
not settle the answer (for example, questions about next week), give your honest best \
estimate. For multi-option questions, give a probability for every option; they must sum to 1."""


def _camel(text: str) -> str:
    return "".join(part.capitalize() for part in text.split("_"))


def build_output_schema(bank: QuestionBank) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for q in bank:
        if q.type == "noul":
            fields[q.id] = (
                float,
                Field(description=f"Probability from 0 to 1 that this is true: {q.instructions}"),
            )
        elif q.type == "choice":
            sub = create_model(
                f"{_camel(q.id)}Probabilities",
                **{
                    label: (float, Field(description=f"Probability that the answer is '{label}': {desc}"))
                    for label, desc in q.criteria.items()
                },
            )
            fields[q.id] = (sub, Field(description=q.instructions))
        else:
            sub = create_model(
                f"{_camel(q.id)}Probabilities",
                **{
                    f"level_{i}": (float, Field(description=f"Probability of level {i}: {desc}"))
                    for i, desc in enumerate(q.criteria)
                },
            )
            fields[q.id] = (sub, Field(description=q.instructions))
    return create_model("AuditAnswers", **fields)


def render_questions(bank: QuestionBank) -> str:
    lines = []
    for q in bank:
        if q.type == "noul":
            lines.append(f"- {q.id} (probability the statement is TRUE): {q.instructions}")
            if q.criteria:
                for key in ("true", "false"):
                    if q.criteria.get(key):
                        lines.append(f"    {key}: {q.criteria[key]}")
        elif q.type == "choice":
            lines.append(f"- {q.id} (one probability per option): {q.instructions}")
            lines += [f"    {label}: {desc}" for label, desc in q.criteria.items()]
        else:
            lines.append(f"- {q.id} (one probability per level, ordered): {q.instructions}")
            lines += [f"    level_{i}: {desc}" for i, desc in enumerate(q.criteria)]
    return "\n".join(lines)


def answers_from_structured(parsed: BaseModel | dict, bank: QuestionBank) -> dict[str, dict]:
    data = parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)
    out: dict[str, dict] = {}
    for q in bank:
        value = data.get(q.id)
        if value is None:
            continue
        if q.type == "noul":
            raw = float(value)
            out[q.id] = noul_answer(raw, raw_p=raw)
        elif q.type == "choice":
            out[q.id] = dist_answer(q, [float(value.get(c, 0.0)) for c in q.classes])
        else:
            out[q.id] = dist_answer(q, [float(value.get(f"level_{i}", 0.0)) for i in range(q.n_classes)])
    return out


class LLMArm(Arm):
    kind = "llm"

    def __init__(self, name, cfg, bank, runnable: Any = None):
        super().__init__(name, cfg, bank)
        if not cfg.model and runnable is None:
            raise ValueError(f"arm {name!r}: set `model`, e.g. provider:model-name")
        self.schema = build_output_schema(bank)
        if runnable is None:
            from langchain.chat_models import init_chat_model

            kwargs: dict[str, Any] = {}
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            llm = init_chat_model(cfg.model, **kwargs)
            # e.g. "json_schema" — some models don't support the
            # forced tool calling that LangChain uses by default.
            method = cfg.extra("structured_method")
            extra = {"method": method} if method else {}
            runnable = llm.with_structured_output(self.schema, include_raw=True, **extra)
        self.runnable = runnable
        self.questions_block = render_questions(bank)
        self.max_retries = int(cfg.extra("max_retries", 2))

    def messages(self, state: StateItem) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage

        user = (
            f"REPORT:\n<<<\n{state.text}\n>>>\n\nQUESTIONS:\n{self.questions_block}\n\nAnswer every question."
        )
        return [SystemMessage(SYSTEM_PROMPT), HumanMessage(user)]

    def predict_one(self, state: StateItem):
        config = {
            "run_name": f"audit:{self.name}",
            "tags": ["jev-calibration-audit", self.name, state.view],
            "metadata": {"state_id": state.state_id, "view": state.view, "arm": self.name},
        }
        msgs = self.messages(state)
        start = time.perf_counter()
        out: dict[str, Any] = {}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                out = self.runnable.invoke(msgs, config=config)
                if out.get("parsed") is not None:
                    break
                last_error = ValueError(f"structured output not parsed: {out.get('parsing_error')}")
            except Exception as exc:  # noqa: BLE001 - provider errors vary by package
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(2.0**attempt)
        latency = time.perf_counter() - start
        if out.get("parsed") is None:
            raise last_error or ValueError("no structured output")

        usage = getattr(out.get("raw"), "usage_metadata", None) or {}
        return self.result(
            state,
            answers=answers_from_structured(out["parsed"], self.bank),
            latency_s=latency,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
