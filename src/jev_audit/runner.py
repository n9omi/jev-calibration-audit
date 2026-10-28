"""Run one arm over every state and append results to a JSONL cache.

Reruns skip states that already have an error-free record for the current question-set
fingerprint, so an interrupted or partially failed run resumes where it stopped and you are
never billed twice for the same call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from jev_audit.arms import Arm, RunContext, make_arm
from jev_audit.config import ExperimentConfig
from jev_audit.data.states import read_states
from jev_audit.questions import QuestionBank


class SetupError(RuntimeError):
    """A missing prerequisite step (e.g. `jev-audit build` has not been run)."""


def predictions_path(cfg: ExperimentConfig, arm_name: str) -> Path:
    return cfg.raw_predictions_dir / f"{arm_name}.jsonl"


def completed_state_ids(path: Path, fingerprint: str) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("question_set") == fingerprint and not rec.get("error"):
                done.add(rec["state_id"])
    return done


def load_labels(cfg: ExperimentConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.data.labels_path, dtype={"week": str, "qid": str})


def run_arm(
    cfg: ExperimentConfig,
    arm_name: str,
    *,
    limit: int | None = None,
    force: bool = False,
    arm: Arm | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    if arm_name not in cfg.arms:
        raise KeyError(f"no arm named {arm_name!r} in config (have: {list(cfg.arms)})")
    if not cfg.data.states_path.exists() or not cfg.data.labels_path.exists():
        raise SetupError(f"no states/labels under {cfg.data.processed_dir}; run `jev-audit build` first")
    bank = QuestionBank.from_yaml(cfg.questions)
    states = read_states(cfg.data.states_path)
    arm = arm or make_arm(arm_name, cfg.arms[arm_name], bank)
    arm.prepare(RunContext(labels=load_labels(cfg)))

    path = predictions_path(cfg, arm_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    fingerprint = bank.fingerprint()
    done = completed_state_ids(path, fingerprint)
    todo = [s for s in states if s.state_id not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"[{arm_name}] {len(done)} cached, {len(todo)} to run (question set {fingerprint})")

    errors = 0
    with path.open("a") as fh:
        for i, res in enumerate(arm.predict_many(todo), start=1):
            rec = res.to_json()
            rec["question_set"] = fingerprint
            rec["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if res.error:
                errors += 1
                if errors <= 3:
                    log(f"[{arm_name}] error on {res.state_id}: {res.error}")
            if i % 50 == 0 or i == len(todo):
                log(f"[{arm_name}] {i}/{len(todo)} done, {errors} errors")
    return path
