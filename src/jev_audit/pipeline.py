"""Pipeline steps shared by the CLI and the tests."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from jev_audit.config import ExperimentConfig
from jev_audit.data.labels import compute_labels
from jev_audit.data.states import build_states, write_states
from jev_audit.evaluate import EvalTables, evaluate
from jev_audit.questions import QuestionBank
from jev_audit.report import write_report
from jev_audit.runner import run_arm


def fetch(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Download (eia) or generate (synthetic) the weekly series and save them as CSV."""
    if cfg.data.source == "synthetic":
        from jev_audit.data.synthetic import synthetic_series

        end = cfg.data.eval_end.isoformat() if cfg.data.eval_end else "2026-10-16"
        frame = synthetic_series(cfg.data.history_start.isoformat(), end, cfg.data.seed)
    else:
        from jev_audit.data.eia import fetch_all

        frame = fetch_all(cfg.eia, start=cfg.data.history_start.isoformat())
    path = cfg.data.series_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="week")
    log(f"wrote {len(frame)} weeks x {frame.shape[1]} series -> {path}")
    return frame


def load_series(cfg: ExperimentConfig) -> pd.DataFrame:
    frame = pd.read_csv(cfg.data.series_csv, parse_dates=["week"], index_col="week")
    return frame.sort_index()


def build(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> None:
    """Build states (what the arms read) and labels (ground truth) from the saved series."""
    bank = QuestionBank.from_yaml(cfg.questions)
    series = load_series(cfg)
    missing = bank.series_used() - set(series.columns)
    if missing:
        log(f"warning: no data for series {sorted(missing)}; questions on them are skipped")
    labels = compute_labels(series, bank)
    cfg.data.labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(cfg.data.labels_path, index=False)
    states = build_states(series, cfg.data)
    write_states(states, cfg.data.states_path)
    by_view = pd.Series([s.view for s in states]).value_counts().to_dict() if states else {}
    log(
        f"wrote {len(labels):,} labels and {len(states):,} states {by_view} "
        f"(question set {bank.fingerprint()})"
    )


def run_enabled(cfg: ExperimentConfig, limit: int | None = None, log: Callable[[str], None] = print):
    for name in cfg.enabled_arms():
        run_arm(cfg, name, limit=limit, log=log)


def evaluate_and_report(cfg: ExperimentConfig, log: Callable[[str], None] = print) -> EvalTables:
    tables = evaluate(cfg)
    tables.save(cfg.results_dir)
    path = write_report(cfg, tables)
    log(f"metrics -> {cfg.results_dir}\nreport  -> {path}")
    return tables
