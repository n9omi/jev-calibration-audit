"""Experiment configuration (YAML -> validated pydantic models) and path resolution."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

View = Literal["with_changes", "levels_only", "wpsr_text"]


class SeriesSpec(BaseModel):
    id: str
    units: Literal["kbbl", "percent"]
    name: str


class DataConfig(BaseModel):
    source: Literal["eia", "synthetic"] = "eia"
    dir: Path = Path("data")
    history_start: date
    eval_start: date
    eval_end: date | None = None
    views: list[View] = Field(default_factory=lambda: ["with_changes", "levels_only"])
    unit_style: Literal["mixed", "million", "thousand"] = "mixed"
    anonymize_dates: bool = True
    seed: int = 7

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.dir / "processed"

    @property
    def series_csv(self) -> Path:
        return self.raw_dir / "series.csv"

    @property
    def wpsr_text_dir(self) -> Path:
        return self.raw_dir / "wpsr_text"

    @property
    def states_path(self) -> Path:
        return self.processed_dir / "states.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.processed_dir / "labels.csv"


class EIAConfig(BaseModel):
    base_url: str = "https://api.eia.gov/v2"
    api_key_env: str = "EIA_API_KEY"
    series: dict[str, SeriesSpec]


class ArmConfig(BaseModel):
    """Per-arm settings. Arm-specific keys (eps, skill, alpha, ...) are allowed as extras."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["jev", "llm", "rules", "climatology", "mock"]
    enabled: bool = True
    model: str | None = None
    max_concurrency: int = 4
    temperature: float | None = None
    price_per_mtok: dict[str, float | None] | None = None

    def extra(self, key: str, default=None):
        return (self.model_extra or {}).get(key, default)


class EvalConfig(BaseModel):
    n_bins: int = 10
    bootstrap_n: int = 2000
    bootstrap_seed: int = 11
    recal_folds: int = 5
    primary_pair: tuple[str, str] | None = None
    reference_arm: str | None = "climatology"


class ExperimentConfig(BaseModel):
    run_name: str
    questions: Path
    results_root: Path = Path("results")
    reports_root: Path = Path("reports")
    data: DataConfig
    eia: EIAConfig
    arms: dict[str, ArmConfig]
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        pair = self.evaluation.primary_pair
        if pair:
            missing = [a for a in pair if a not in self.arms]
            if missing:
                raise ValueError(f"evaluation.primary_pair names unknown arms: {missing}")
        ref = self.evaluation.reference_arm
        if ref and ref not in self.arms:
            raise ValueError(f"evaluation.reference_arm {ref!r} is not a configured arm")
        return self

    # ------------------------------------------------------------------ paths
    @property
    def results_dir(self) -> Path:
        return self.results_root / self.run_name

    @property
    def raw_predictions_dir(self) -> Path:
        return self.results_dir / "raw"

    @property
    def report_dir(self) -> Path:
        return self.reports_root / self.run_name

    def enabled_arms(self) -> list[str]:
        return [name for name, arm in self.arms.items() if arm.enabled]

    def resolve(self, repo_root: Path, out_root: Path | None = None) -> ExperimentConfig:
        """Return a copy with absolute paths.

        `questions` resolves against the repository; data, results and reports resolve against
        `out_root` (defaults to the repository), which lets tests write into a temp folder.
        """
        out = out_root or repo_root

        def _abs(p: Path, base: Path) -> Path:
            return p if p.is_absolute() else (base / p)

        data = self.data.model_copy(update={"dir": _abs(self.data.dir, out)})
        return self.model_copy(
            update={
                "questions": _abs(self.questions, repo_root),
                "results_root": _abs(self.results_root, out),
                "reports_root": _abs(self.reports_root, out),
                "data": data,
            }
        )


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the folder holding pyproject.toml (falls back to `start`)."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    return start


def load_config(path: str | Path, out_root: str | Path | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    cfg = ExperimentConfig.model_validate(raw)
    repo_root = find_repo_root(path.parent)
    return cfg.resolve(repo_root, Path(out_root).resolve() if out_root else None)
