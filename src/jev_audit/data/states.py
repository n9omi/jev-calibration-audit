"""Build the text each arm classifies (the "state") for every report week and view.

Views
-----
with_changes  Templated summary that states each week-over-week change and the new level.
              Horizon-0 questions are mostly extraction.
levels_only   Templated summary with this week's and last week's levels but no changes.
              Horizon-0 questions now need subtraction / percentage arithmetic.
wpsr_text     Real EIA Weekly Petroleum Status Report highlights you supply as
              data/raw/wpsr_text/YYYY-MM-DD.txt (week-ending date). Natural phrasing,
              changes rounded to 0.1 million barrels.

Templated text is phrased several ways (seeded per state) and written in either million or
thousand barrels, so a model cannot key on one sentence shape. Numbers are exact (million
barrels use three decimals) so every horizon-0 label is fully determined by the text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jev_audit.config import DataConfig
from jev_audit.data.labels import regular_weekly

PRODUCT_ORDER = ["crude", "cushing", "gasoline", "distillate"]

SUBJECTS = {
    "crude": [
        "U.S. commercial crude oil inventories (excluding the Strategic Petroleum Reserve)",
        "Commercial crude oil stocks excluding the SPR",
    ],
    "cushing": ["Crude oil stocks at Cushing, Oklahoma", "Cushing, Oklahoma crude inventories"],
    "gasoline": ["Total motor gasoline inventories", "Gasoline stocks"],
    "distillate": ["Distillate fuel inventories", "Distillate stocks"],
}

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


@dataclass
class StateItem:
    state_id: str
    week: str
    view: str
    text: str
    source: str

    def to_json(self) -> dict:
        return asdict(self)


def fmt_volume(kbbl: float, style: str) -> str:
    if style == "million":
        return f"{kbbl / 1000:.3f} million barrels"
    return f"{int(round(kbbl)):,} thousand barrels"


def _render_change(subject: str, level: float, change: float, style: str, rng) -> str:
    if change == 0:
        return f"{subject} were unchanged from the previous week at {fmt_volume(level, style)}."
    if rng.integers(2) == 0:
        verb = "rose" if change > 0 else "fell"
        return (
            f"{subject} {verb} by {fmt_volume(abs(change), style)} from the previous week "
            f"to {fmt_volume(level, style)}."
        )
    verb = "increased" if change > 0 else "decreased"
    return (
        f"{subject} {verb} {fmt_volume(abs(change), style)} week over week, "
        f"ending at {fmt_volume(level, style)}."
    )


def _render_levels(subject: str, level: float, prev: float, style: str, rng) -> str:
    if rng.integers(2) == 0:
        return (
            f"{subject} stood at {fmt_volume(level, style)}, compared with "
            f"{fmt_volume(prev, style)} the previous week."
        )
    return f"{subject} totaled {fmt_volume(level, style)} versus {fmt_volume(prev, style)} a week earlier."


def _render_util(util: float, rng) -> str:
    if rng.integers(2) == 0:
        return f"Refineries operated at {util:.1f}% of their operable capacity."
    return f"Refinery utilization was {util:.1f}% of operable capacity."


def render_state(
    row: pd.Series,
    prev: pd.Series,
    view: str,
    rng: np.random.Generator,
    unit_style: str,
    week: pd.Timestamp | None = None,
) -> str:
    style = unit_style if unit_style != "mixed" else ("million", "thousand")[rng.integers(2)]
    lines = []
    if week is not None:
        lines.append(f"Report for the week ending {week.strftime('%B')} {week.day}, {week.year}.")
    else:
        lines.append("Weekly U.S. petroleum inventory summary.")
    for product in PRODUCT_ORDER:
        if product not in row.index or pd.isna(row[product]) or pd.isna(prev[product]):
            continue
        subject = SUBJECTS[product][rng.integers(len(SUBJECTS[product]))]
        level, before = float(row[product]), float(prev[product])
        if view == "with_changes":
            lines.append(_render_change(subject, level, level - before, style, rng))
        else:
            lines.append(_render_levels(subject, level, before, style, rng))
    if "refinery_util" in row.index and not pd.isna(row["refinery_util"]):
        lines.append(_render_util(float(row["refinery_util"]), rng))
    return " ".join(lines)


def scrub_dates(text: str) -> str:
    text = re.sub(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", "[YEAR]", text)
    return text


def load_wpsr_text(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {p.stem: p.read_text().strip() for p in sorted(directory.glob("*.txt"))}


def _state_rng(seed: int, week: pd.Timestamp, view: str) -> np.random.Generator:
    # Deterministic per (seed, week, view), independent of iteration order.
    return np.random.default_rng([seed, week.toordinal(), sum(map(ord, view))])


def build_states(series: pd.DataFrame, data: DataConfig) -> list[StateItem]:
    s = regular_weekly(series)
    start = pd.Timestamp(data.eval_start)
    end = pd.Timestamp(data.eval_end) if data.eval_end else s.index.max()
    wpsr = load_wpsr_text(data.wpsr_text_dir) if "wpsr_text" in data.views else {}
    items: list[StateItem] = []
    for i in range(1, len(s)):
        week = s.index[i]
        if week < start or week > end:
            continue
        row, prev = s.iloc[i], s.iloc[i - 1]
        if row.isna().all() or prev.isna().all():
            continue
        iso = week.date().isoformat()
        for view in data.views:
            if view == "wpsr_text":
                if iso not in wpsr:
                    continue
                text = scrub_dates(wpsr[iso]) if data.anonymize_dates else wpsr[iso]
                source = "eia_wpsr"
            else:
                rng = _state_rng(data.seed, week, view)
                text = render_state(
                    row,
                    prev,
                    view,
                    rng,
                    data.unit_style,
                    week=None if data.anonymize_dates else week,
                )
                source = "templated"
            items.append(StateItem(f"{iso}|{view}", iso, view, text, source))
    return items


def write_states(items: Iterable[StateItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item.to_json()) + "\n")


def read_states(path: Path) -> list[StateItem]:
    with path.open() as fh:
        return [StateItem(**json.loads(line)) for line in fh if line.strip()]
