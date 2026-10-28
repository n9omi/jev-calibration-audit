"""Classifier arms. Each arm answers the same question bank about the same states."""

from __future__ import annotations

from typing import Any

from jev_audit.arms.base import Arm, ArmResult, RunContext
from jev_audit.config import ArmConfig
from jev_audit.questions import QuestionBank


def make_arm(name: str, cfg: ArmConfig, bank: QuestionBank, **kwargs: Any) -> Arm:
    if cfg.kind == "jev":
        from jev_audit.arms.jev import JevArm

        return JevArm(name, cfg, bank, **kwargs)
    if cfg.kind == "llm":
        from jev_audit.arms.llm import LLMArm

        return LLMArm(name, cfg, bank, **kwargs)
    if cfg.kind == "rules":
        from jev_audit.arms.rules import RulesArm

        return RulesArm(name, cfg, bank)
    if cfg.kind == "climatology":
        from jev_audit.arms.climatology import ClimatologyArm

        return ClimatologyArm(name, cfg, bank)
    if cfg.kind == "mock":
        from jev_audit.arms.mock import MockArm

        return MockArm(name, cfg, bank)
    raise ValueError(f"unknown arm kind {cfg.kind!r}")


__all__ = ["Arm", "ArmResult", "RunContext", "make_arm"]
