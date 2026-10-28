"""Deterministic parser for inventory summaries (templated text and real WPSR highlights).

Used by the rules baseline. It answers the question "how much of horizon 0 is just
extraction?" — anything it gets right needs no model at all.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(million|thousand)\s+barrels", re.IGNORECASE)
_UP = re.compile(r"\b(rose|increased|built)\b")
_DOWN = re.compile(r"\b(fell|decreased|drew)\b")
_UNCHANGED = re.compile(r"\bunchanged\b")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _kbbl(number: str, unit: str) -> float:
    value = float(number.replace(",", ""))
    return round(value * 1000.0, 3) if unit.lower() == "million" else value


def _sentences(text: str) -> list[str]:
    text = text.replace("U.S.", "US")
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _product(sentence: str) -> str | None:
    low = sentence.lower()
    if "%" in sentence and ("operable capacity" in low or "utilization" in low):
        return "refinery_util"
    if "per day" in low or " a day" in low:
        return None  # flows (b/d), not stocks
    if "cushing" in low:
        return "cushing"
    if "gasoline" in low:
        return "gasoline"
    if "distillate" in low:
        return "distillate"
    if "crude" in low:
        return "crude"
    return None


def parse_report(text: str) -> dict[str, dict[str, float]]:
    """Return {product: {"change", "level", "prev"}} in kbbl, and {"refinery_util": {"level"}}.

    Keys are present only when the text supports them.
    """
    out: dict[str, dict[str, float]] = {}
    for sentence in _sentences(text):
        product = _product(sentence)
        if product is None:
            continue
        found = out.setdefault(product, {})
        low = sentence.lower()
        if product == "refinery_util":
            m = _PCT.search(sentence)
            if m:
                found["level"] = float(m.group(1))
            continue
        nums = [_kbbl(n, u) for n, u in _NUM.findall(sentence)]
        if not nums:
            continue
        if ("compared with" in low or "versus" in low) and len(nums) >= 2:
            found["level"], found["prev"] = nums[0], nums[1]
        elif _UNCHANGED.search(low):
            found["change"] = 0.0
            found["level"] = nums[-1]
        elif _UP.search(low) or _DOWN.search(low):
            sign = 1.0 if _UP.search(low) else -1.0
            found["change"] = sign * nums[0]
            if len(nums) >= 2:
                found["level"] = nums[1]
        elif low.startswith("at "):
            found.setdefault("level", nums[0])  # WPSR: "At 421.7 million barrels, US crude ..."
    for found in out.values():
        if "change" not in found and {"level", "prev"} <= found.keys():
            found["change"] = found["level"] - found["prev"]
        if "prev" not in found and {"level", "change"} <= found.keys():
            found["prev"] = found["level"] - found["change"]
    return {k: v for k, v in out.items() if v}
