from datetime import date

import numpy as np
import pandas as pd
import pytest

from jev_audit.arms.rules import deterministic_answer
from jev_audit.config import DataConfig
from jev_audit.data.eia import rows_to_series
from jev_audit.data.labels import compute_labels, regular_weekly
from jev_audit.data.parse import parse_report
from jev_audit.data.states import build_states, render_state, scrub_dates
from jev_audit.data.synthetic import synthetic_series


def _series():
    weeks = pd.date_range("2024-01-05", periods=5, freq="7D")
    return pd.DataFrame(
        {
            "crude": [400_000, 398_000, 398_500, 392_000, 392_000],
            "gasoline": [230_000, 231_000, 229_000, 229_500, 228_000],
            "distillate": [120_000, 119_000, 125_000, 124_000, 124_500],
            "cushing": [30_000, 31_000, 30_500, 30_400, 30_600],
            "refinery_util": [88.2, 91.0, 95.3, 84.9, 90.0],
        },
        index=weeks,
    )


def _labels(bank):
    lab = compute_labels(_series(), bank)
    return {(r.week, r.qid): r.y for r in lab.itertuples()}


def test_draw_and_lead_labels(bank):
    y = _labels(bank)
    assert y[("2024-01-12", "crude_draw")] == 1  # 400,000 -> 398,000
    assert y[("2024-01-19", "crude_draw")] == 0  # build of 500
    assert y[("2024-01-12", "next_crude_draw")] == 0  # next week's change is the +500 build
    assert ("2024-02-02", "crude_draw") in y
    assert y[("2024-02-02", "crude_draw")] == 0  # unchanged counts as "not a draw"
    assert ("2024-02-02", "next_crude_draw") not in y  # no next week yet
    assert ("2024-01-05", "crude_draw") not in y  # no previous week


def test_bucket_labels(bank):
    y = _labels(bank)
    assert y[("2024-01-12", "crude_change_bucket")] == 1  # -2,000 kbbl -> draw of 1-5M
    assert y[("2024-01-26", "crude_change_bucket")] == 0  # -6,500 kbbl -> draw > 5M
    assert y[("2024-01-19", "crude_change_bucket")] == 2  # +500 -> small
    assert y[("2024-01-19", "refinery_util_bucket")] == 3  # 95.3% -> 95% or higher
    assert y[("2024-01-26", "refinery_util_bucket")] == 0  # 84.9% -> below 85%


def test_largest_pct_mover(bank):
    y = _labels(bank)
    # 2024-01-19: crude +0.13%, gasoline -0.87%, distillate +5.04% -> distillate (index 2)
    assert y[("2024-01-19", "largest_pct_mover")] == 2


def test_missing_week_breaks_the_diff(bank):
    s = _series().drop(pd.Timestamp("2024-01-19"))
    lab = compute_labels(s, bank)
    assert not ((lab["week"] == "2024-01-26") & (lab["qid"] == "crude_draw")).any()
    assert regular_weekly(s)["crude"].isna().sum() == 1


@pytest.mark.parametrize("view", ["with_changes", "levels_only"])
@pytest.mark.parametrize("style", ["million", "thousand"])
def test_parse_inverts_render_exactly(view, style):
    s = synthetic_series("2020-01-03", "2021-12-31", seed=11)
    for i in range(1, len(s)):
        rng = np.random.default_rng(i)
        text = render_state(s.iloc[i], s.iloc[i - 1], view, rng, style)
        parsed = parse_report(text)
        for product in ["crude", "gasoline", "distillate", "cushing"]:
            change = s.iloc[i][product] - s.iloc[i - 1][product]
            assert parsed[product]["change"] == pytest.approx(change, abs=1e-6), text
            assert parsed[product]["level"] == pytest.approx(s.iloc[i][product], abs=1e-6)
        assert parsed["refinery_util"]["level"] == pytest.approx(s.iloc[i]["refinery_util"])


def test_rules_answers_match_labels_on_templated_text(bank):
    s = synthetic_series("2020-01-03", "2021-12-31", seed=3)
    labels = compute_labels(s, bank)
    truth = {(r.week, r.qid): r.y for r in labels.itertuples()}
    data = DataConfig(history_start=date(2020, 1, 3), eval_start=date(2020, 1, 10), seed=1)
    states = build_states(s, data)
    checked = 0
    for st in states:
        parsed = parse_report(st.text)
        for q in bank:
            if q.horizon != 0 or (st.week, q.id) not in truth:
                continue
            assert deterministic_answer(q, parsed) == truth[(st.week, q.id)], (q.id, st.text)
            checked += 1
    assert checked > 1000


def test_parse_real_wpsr_phrasing():
    text = (
        "U.S. crude oil refinery inputs averaged 16.2 million barrels per day during the week "
        "ending October 17, 2025, which was 241,000 barrels per day more than the previous "
        "week's average. Refineries operated at 92.3% of their operable capacity last week. "
        "U.S. commercial crude oil inventories (excluding those in the Strategic Petroleum "
        "Reserve) decreased by 2.6 million barrels from the previous week. At 421.7 million "
        "barrels, U.S. crude oil inventories are about 4% below the five year average for this "
        "time of year. Total motor gasoline inventories increased by 1.2 million barrels from "
        "last week and are about 3% below the five year average. Distillate fuel inventories "
        "remained unchanged last week at 115.0 million barrels."
    )
    p = parse_report(text)
    assert p["refinery_util"]["level"] == 92.3
    assert p["crude"]["change"] == -2600
    assert p["crude"]["level"] == 421_700
    assert p["gasoline"]["change"] == 1200
    assert p["distillate"]["change"] == 0
    assert "cushing" not in p


def test_scrub_dates():
    out = scrub_dates("For the week ending October 17, 2025 (10/17/2025), stocks rose in 2025.")
    assert "October" not in out and "2025" not in out and "10/17" not in out


def test_states_hide_dates_by_default():
    s = synthetic_series("2023-01-06", "2023-03-31", seed=1)
    data = DataConfig(history_start=date(2023, 1, 6), eval_start=date(2023, 1, 13))
    states = build_states(s, data)
    assert states and all("2023" not in st.text for st in states)
    assert {st.view for st in states} == {"with_changes", "levels_only"}
    # deterministic: same inputs -> same text
    assert [st.text for st in build_states(s, data)] == [st.text for st in states]


def test_eia_rows_to_series():
    rows = [
        {"period": "2024-01-12", "series": "WCESTUS1", "value": "421000", "units": "MBBL"},
        {"period": "2024-01-05", "series": "WCESTUS1", "value": 420500, "units": "MBBL"},
        {"period": "2024-01-19", "series": "WCESTUS1", "value": None},
    ]
    s = rows_to_series(rows)
    assert list(s.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert s.iloc[-1] == 421_000.0
