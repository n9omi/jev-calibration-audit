import json

import httpx2
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from jev_audit.arms.base import RunContext
from jev_audit.arms.climatology import ClimatologyArm
from jev_audit.arms.jev import JevArm
from jev_audit.arms.llm import LLMArm, build_output_schema
from jev_audit.config import ArmConfig
from jev_audit.data.states import StateItem

STATE = StateItem("2024-01-12|with_changes", "2024-01-12", "with_changes", "Some report text.", "templated")


def _fake_typesafe_answers(bank, payload):
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.8}
        elif q["type"] == "choice":
            labels = list(q["criteria"])
            probs = {label: 1 / len(labels) for label in labels}
            answers[qid] = {"type": "choice", "choice": labels[0], "probabilities": probs, "confidence": 0.1}
        else:
            k = len(q["criteria"])
            probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(k)}
            answers[qid] = {
                "type": "score",
                "score": 1.0,
                "probabilities": probs,
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "confidence": 0.9,
            }
    return {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 321, "output_tokens": 0}}


def test_jev_arm_sends_one_request_with_all_questions(bank, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        seen.append((request, payload))
        return httpx2.Response(200, json=_fake_typesafe_answers(bank, payload))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    arm = JevArm("jev", ArmConfig(kind="jev", model="jev-latest"), bank, client=client)
    res = arm.predict_one(STATE)

    assert len(seen) == 1
    request, payload = seen[0]
    assert str(request.url).endswith("/v1/systemone")
    assert request.headers["authorization"] == "Bearer test-key"
    assert payload["model"] == "jev-latest"
    assert payload["state"] == "Some report text."
    assert set(payload["questions"]) == set(bank.ids())
    assert payload["questions"]["crude_draw"]["criteria"]["true"].startswith("Commercial crude")
    assert payload["questions"]["refinery_util_bucket"]["type"] == "score"

    assert res.error is None and res.model == "jev-test" and res.input_tokens == 321
    assert res.answers["crude_draw"] == {"type": "noul", "p": 0.8}
    score = res.answers["crude_change_bucket"]
    assert score["probs"] == [0.0, 1.0, 0.0, 0.0, 0.0] and score["confidence"] == 0.9
    choice = res.answers["largest_pct_mover"]
    assert choice["probs"] == pytest.approx([1 / 3] * 3)


def test_jev_arm_retries_rate_limit(bank, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"error": "slow down"})
        return httpx2.Response(200, json=_fake_typesafe_answers(bank, json.loads(request.content)))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    arm = JevArm("jev", ArmConfig(kind="jev"), bank, client=client)
    res = arm.predict_one(STATE)
    assert calls["n"] == 2 and res.error is None


def test_jev_arm_records_error_instead_of_crashing(bank, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    client = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(401, json={})))
    arm = JevArm("jev", ArmConfig(kind="jev"), bank, client=client)
    [res] = list(arm.predict_many([STATE], max_concurrency=1))
    assert res.error and "401" in res.error


def test_llm_arm_parses_structured_output(bank):
    schema = build_output_schema(bank)
    payload = {}
    for q in bank:
        if q.type == "noul":
            payload[q.id] = 1.3  # out of range on purpose -> clipped
        elif q.type == "choice":
            payload[q.id] = {c: 1.0 for c in q.classes}  # sums to 3 -> renormalized
        else:
            payload[q.id] = {f"level_{i}": (2.0 if i == 0 else 0.0) for i in range(q.n_classes)}
    parsed = schema.model_validate(payload)
    seen = {}

    def fake(messages, config=None):
        seen["messages"] = messages
        raw = AIMessage(
            content="", usage_metadata={"input_tokens": 900, "output_tokens": 150, "total_tokens": 1050}
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}

    arm = LLMArm("llm", ArmConfig(kind="llm", model="fake:model"), bank, runnable=RunnableLambda(fake))
    res = arm.predict_one(STATE)
    assert res.error is None and res.input_tokens == 900 and res.output_tokens == 150
    assert res.answers["crude_draw"]["p"] == 1.0 and res.answers["crude_draw"]["raw_p"] == 1.3
    assert res.answers["largest_pct_mover"]["probs"] == pytest.approx([1 / 3] * 3)
    assert res.answers["largest_pct_mover"]["raw_sum"] == pytest.approx(3.0)
    assert res.answers["refinery_util_bucket"]["probs"][0] == pytest.approx(1.0)
    prompt = seen["messages"][1].content
    assert "Some report text." in prompt
    for q in bank:  # same instructions as the Jev arm, word for word
        assert q.instructions in prompt


def test_climatology_uses_only_past_weeks(bank):
    labels = pd.DataFrame(
        {
            "week": ["2024-01-05", "2024-01-12", "2024-01-19"],
            "qid": ["crude_draw"] * 3,
            "y": [1, 1, 0],
        }
    )
    arm = ClimatologyArm("clim", ArmConfig(kind="climatology", alpha=1.0), bank)
    arm.prepare(RunContext(labels=labels))
    # predicting 2024-01-12: only 2024-01-05 (y=1) is visible -> (1 + 1) / (1 + 2)
    assert arm.dist("crude_draw", pd.Timestamp("2024-01-12"))[1] == pytest.approx(2 / 3)
    # predicting 2024-01-26: all three visible -> (2 + 1) / (3 + 2)
    assert arm.dist("crude_draw", pd.Timestamp("2024-01-26"))[1] == pytest.approx(3 / 5)
    assert np.allclose(arm.dist("crude_draw", pd.Timestamp("2023-01-01")), [0.5, 0.5])
