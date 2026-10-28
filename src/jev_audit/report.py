"""Render evaluation tables and figures into a markdown report."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from jev_audit.config import ExperimentConfig
from jev_audit.evaluate import EvalTables
from jev_audit.plots import pairwise_figure, reliability_figure


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != 0 and abs(v) < 10 ** (-digits):
            return f"{v:.1e}"
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    cols = [c for c in cols if c in df.columns]
    headers = headers or cols
    lines = ["| " + " | ".join(headers[: len(cols)]) + " |", "|" + "---|" * len(cols)]
    for row in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _arm_order(cfg: ExperimentConfig, arms: list[str]) -> list[str]:
    first = list(cfg.evaluation.primary_pair or [])
    rest = [a for a in cfg.arms if a not in first]
    return [a for a in first + rest if a in arms]


HOW_TO_READ = """\
- **Brier** (Noul): mean squared error of the probability; lower is better. Always guessing
  0.5 scores 0.25. **BSS** is skill vs the climatology arm: >0 beats the base rate.
- **ECE**: count-weighted average gap between forecast probability and observed frequency
  across 10 bins (lower is better). `ece_pooled` bins all Noul answers in the group together.
- **Murphy decomposition**: Brier = reliability − resolution + uncertainty. Reliability is the
  calibration penalty; resolution is how well forecasts separate outcomes.
- **Spiegelhalter Z**: test of H0 "these probabilities are calibrated"; small p rejects it.
- **RPS** (Score questions): ranked probability score, the proper score for ordered levels;
  mass on a neighbouring level costs less than mass far away. **mbrier** (Choice): multiclass
  Brier, 0 to 2.
- **Paired comparisons**: mean loss difference on identical items (A − B; negative = A better),
  95% circular block-bootstrap CI, and a Diebold–Mariano test with Newey–West variance,
  because weekly losses are autocorrelated.
- **Recalibration**: out-of-time Platt / isotonic maps fitted on earlier weeks only. If ECE
  drops a lot, the model ranks well but its probabilities need a correction layer.
"""

CAVEATS = """\
- **Training-data contamination.** Historical EIA reports are public and may be in any model's
  training data. Dates are stripped from the text, but levels can still identify a week. The
  cleanest evidence is the prospective track: weeks published after the model's release.
- **Templated text.** The `with_changes` / `levels_only` views are generated from EIA numbers
  so labels are exact; real report prose (`wpsr_text`) is the external-validity check.
- **Verbalized LLM probabilities.** The LLM arm writes its probabilities down. Token
  log-probabilities can calibrate differently; that is a separate experiment.
- **Jev `confidence`** is a concentration measure, not the chosen label's probability, so it is
  evaluated only as an abstention signal (`confidence_auroc`), never as a probability.
- **Model drift.** `jev-latest` and provider aliases can change under you. Pin model ids for
  any result you publish and record the run date.
"""


def write_report(cfg: ExperimentConfig, t: EvalTables) -> Path:
    out = cfg.report_dir
    figs = out / "figures"
    arms = _arm_order(cfg, t.meta["arms"])
    m = t.meta
    parts: list[str] = [f"# Jev calibration audit — `{cfg.run_name}`\n"]
    if m.get("synthetic"):
        parts.append(
            "> **SYNTHETIC DEMO.** Generated inventory series and simulated forecasters "
            "(`mock_*` arms read the true labels). This shows the pipeline works; it says "
            "nothing about Jev or any real model.\n"
        )
    parts.append(
        f"Generated {date.today().isoformat()} · data `{m['data_source']}` · question set "
        f"`{m['question_set']}` · report weeks {m['weeks'][0]} to {m['weeks'][1]} · views "
        f"{', '.join(m['views'])} · **{m['n_common_items']:,} items scored by every arm** "
        f"({m['n_states']:,} states).\n"
    )

    parts.append("## Headline\n")
    s = t.summary.copy()
    if not s.empty:
        s["arm"] = pd.Categorical(s["arm"], categories=arms, ordered=True)
        s = s.sort_values(["horizon", "view", "arm"])
    parts.append(
        md_table(
            s,
            [
                "arm",
                "view",
                "horizon",
                "noul_brier",
                "noul_ece",
                "noul_ece_pooled",
                "noul_accuracy",
                "score_rps",
                "choice_mbrier",
            ],
            [
                "arm",
                "view",
                "h",
                "Noul Brier",
                "Noul ECE",
                "ECE pooled",
                "Noul acc.",
                "Score RPS",
                "Choice mBrier",
            ],
        )
    )

    fig = reliability_figure(
        t.reliability,
        arms,
        figs / "reliability.png",
        "Reliability of Noul probabilities (pooled across questions)",
    )
    if fig:
        parts.append("\n## Reliability\n\n![Reliability diagrams](figures/reliability.png)\n")

    parts.append("\n## Paired comparisons\n")
    pw = t.pairwise
    if pw.empty:
        parts.append("_No paired comparisons (need at least two arms on common items)._\n")
    else:
        for (a, b), g in pw.groupby(["arm_a", "arm_b"], sort=False):
            parts.append(f"\n### `{a}` vs `{b}`\n")
            parts.append(
                md_table(
                    g.sort_values(["horizon", "qid", "view"]),
                    [
                        "qid",
                        "view",
                        "horizon",
                        "loss",
                        "n",
                        "loss_a",
                        "loss_b",
                        "diff",
                        "ci_lo",
                        "ci_hi",
                        "dm_stat",
                        "dm_p",
                    ],
                    [
                        "question",
                        "view",
                        "h",
                        "loss",
                        "n",
                        f"{a}",
                        f"{b}",
                        "diff (A−B)",
                        "CI low",
                        "CI high",
                        "DM stat",
                        "DM p",
                    ],
                )
            )
            if cfg.evaluation.primary_pair and (a, b) == tuple(cfg.evaluation.primary_pair):
                f = pairwise_figure(pw, a, b, figs / "pairwise.png")
                if f:
                    parts.append("\n![Paired loss differences](figures/pairwise.png)\n")

    parts.append("\n## Per-question detail: Noul\n")
    b = t.binary.copy()
    if not b.empty:
        b["arm"] = pd.Categorical(b["arm"], categories=arms, ordered=True)
        b = b.sort_values(["horizon", "qid", "view", "arm"])
    parts.append(
        md_table(
            b,
            [
                "qid",
                "view",
                "arm",
                "n",
                "base_rate",
                "mean_p",
                "accuracy",
                "brier",
                "bss",
                "log_loss",
                "ece",
                "rel",
                "res",
                "auroc",
                "spiegelhalter_p",
            ],
            [
                "question",
                "view",
                "arm",
                "n",
                "base rate",
                "mean p",
                "acc.",
                "Brier",
                "BSS",
                "log loss",
                "ECE",
                "REL",
                "RES",
                "AUROC",
                "Spiegelhalter p",
            ],
        )
    )
    parts.append("\n## Per-question detail: Choice and Score\n")
    mc = t.multiclass.copy()
    if not mc.empty:
        mc["arm"] = pd.Categorical(mc["arm"], categories=arms, ordered=True)
        mc = mc.sort_values(["horizon", "qid", "view", "arm"])
    parts.append(
        md_table(
            mc,
            [
                "qid",
                "type",
                "view",
                "arm",
                "n",
                "accuracy",
                "mbrier",
                "rps",
                "level_mae",
                "nll",
                "top_ece",
                "classwise_ece",
                "maxprob_auroc",
                "confidence_auroc",
            ],
            [
                "question",
                "type",
                "view",
                "arm",
                "n",
                "acc.",
                "mBrier",
                "RPS",
                "level MAE",
                "NLL",
                "top-label ECE",
                "classwise ECE",
                "max-prob AUROC",
                "Jev confidence AUROC",
            ],
        )
    )

    parts.append("\n## Is the miscalibration fixable?\n")
    rc = t.recal.copy()
    if not rc.empty and "ece_raw" in rc:
        rc = rc.dropna(subset=["ece_raw"])
    parts.append(
        md_table(
            rc,
            [
                "qid",
                "view",
                "arm",
                "n_eval",
                "ece_raw",
                "ece_platt",
                "ece_isotonic",
                "brier_raw",
                "brier_platt",
                "brier_isotonic",
            ],
            [
                "question",
                "view",
                "arm",
                "n (held-out)",
                "ECE raw",
                "ECE Platt",
                "ECE isotonic",
                "Brier raw",
                "Brier Platt",
                "Brier isotonic",
            ],
        )
    )

    parts.append("\n## Cost, latency and coverage\n")
    parts.append(
        md_table(
            t.ops,
            [
                "arm",
                "kind",
                "model",
                "calls",
                "errors",
                "coverage",
                "latency_median_s",
                "latency_p95_s",
                "input_tokens_mean",
                "output_tokens_mean",
                "usd_per_1k_states",
            ],
            [
                "arm",
                "kind",
                "model",
                "calls",
                "errors",
                "coverage",
                "latency p50 (s)",
                "latency p95 (s)",
                "input tok",
                "output tok",
                "USD / 1k states",
            ],
        )
    )
    parts.append(
        "\nLatency is wall-clock per call at the configured concurrency; cost uses the prices in "
        "the config, not live pricing.\n"
    )
    parts.append("\n## How to read this\n\n" + HOW_TO_READ)
    parts.append("\n## Caveats\n\n" + CAVEATS)

    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text("\n".join(parts))
    return path
