"""Question bank: typed questions shared by every arm, plus their ground-truth label specs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


class LabelSpec(BaseModel):
    """How the ground truth for a question is computed from the EIA series.

    fn:
      draw                  -> y = 1 if the week-over-week change is negative
      bucket                -> y = index of the bucket the value falls in (see `edges`)
      argmax_abs_pct_change -> y = index of the series with the largest |% change|
    lead: 0 = this week's report, 1 = next week's report.
    """

    fn: Literal["draw", "bucket", "argmax_abs_pct_change"]
    series: str | list[str]
    lead: int = 0
    transform: Literal["diff", "level"] = "diff"
    edges: list[float] | None = None

    @property
    def series_list(self) -> list[str]:
        return [self.series] if isinstance(self.series, str) else list(self.series)


class QuestionSpec(BaseModel):
    id: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, Any] | list[Any] | None = None
    label: LabelSpec

    @field_validator("criteria", mode="before")
    @classmethod
    def _yaml_bool_keys(cls, value):
        # YAML reads bare `true:` / `false:` keys as booleans.
        if isinstance(value, dict):
            return {(str(k).lower() if isinstance(k, bool) else k): v for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _validate(self) -> QuestionSpec:
        if not _IDENT.match(self.id):
            raise ValueError(f"question id {self.id!r} must be snake_case")
        lab = self.label
        if self.type == "noul":
            if lab.fn != "draw":
                raise ValueError(f"{self.id}: noul questions use label fn 'draw'")
            if self.criteria is not None and not (
                isinstance(self.criteria, dict) and set(self.criteria) <= {"true", "false"}
            ):
                raise ValueError(f"{self.id}: noul criteria must be a mapping with true/false")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError(f"{self.id}: choice criteria must map >= 2 labels to text")
            bad = [k for k in self.criteria if not _IDENT.match(k)]
            if bad:
                raise ValueError(f"{self.id}: choice labels must be snake_case: {bad}")
            if lab.fn == "argmax_abs_pct_change" and list(self.criteria) != lab.series_list:
                raise ValueError(f"{self.id}: choice labels must equal label.series, in order")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError(f"{self.id}: score criteria must be a list of >= 2 levels")
            if lab.fn == "bucket" and (lab.edges is None or len(lab.edges) + 1 != len(self.criteria)):
                raise ValueError(f"{self.id}: score needs len(edges) + 1 == number of levels")
        if lab.fn == "bucket" and (not lab.edges or sorted(lab.edges) != lab.edges):
            raise ValueError(f"{self.id}: bucket edges must be non-empty and increasing")
        return self

    # ----------------------------------------------------------------- helpers
    @property
    def horizon(self) -> int:
        return self.label.lead

    @property
    def classes(self) -> list[str]:
        """Class names in a fixed order. Labels (y) are indices into this list."""
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "choice":
            return list(self.criteria)  # type: ignore[arg-type]
        return [str(level) for level in self.criteria]  # type: ignore[union-attr]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


class QuestionBank:
    def __init__(self, questions: dict[str, QuestionSpec]):
        self.questions = questions

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuestionBank:
        with Path(path).open() as fh:
            raw = yaml.safe_load(fh)
        qs = {qid: QuestionSpec(id=qid, **spec) for qid, spec in raw["questions"].items()}
        return cls(qs)

    def __iter__(self) -> Iterator[QuestionSpec]:
        return iter(self.questions.values())

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, qid: str) -> QuestionSpec:
        return self.questions[qid]

    def ids(self) -> list[str]:
        return list(self.questions)

    def series_used(self) -> set[str]:
        return {s for q in self for s in q.label.series_list}

    def fingerprint(self) -> str:
        """Short hash of the exact wording; cached predictions are keyed on it."""
        payload = {
            qid: q.model_dump(mode="json", exclude={"id"}) for qid, q in sorted(self.questions.items())
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_typesafe(self) -> dict[str, Any]:
        """Build langchain_typesafe Noul / Choice / Score objects (lazy import)."""
        from langchain_typesafe import Choice, Noul, NoulCriteria, Score

        out: dict[str, Any] = {}
        for q in self:
            if q.type == "noul":
                crit = NoulCriteria(**q.criteria) if q.criteria else None
                out[q.id] = Noul(instructions=q.instructions, criteria=crit)
            elif q.type == "choice":
                out[q.id] = Choice(instructions=q.instructions, criteria=dict(q.criteria))
            else:
                out[q.id] = Score(instructions=q.instructions, criteria=list(q.criteria))
        return out
