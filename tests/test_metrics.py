import numpy as np
import pytest

from jev_audit.metrics.binary import (
    auroc,
    brier,
    ece,
    log_loss,
    murphy_decomposition,
    reliability_table,
    spiegelhalter,
)
from jev_audit.metrics.compare import block_bootstrap_ci, compare_losses, newey_west_var
from jev_audit.metrics.multiclass import multiclass_brier_per, rps_per, top_label_ece
from jev_audit.metrics.recalibrate import recalibration_report


def _calibrated(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return y, p


def test_brier_and_log_loss_known_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.5])
    assert brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.25) / 4)
    expected = -np.mean(np.log([0.9, 0.8, 0.6, 0.5]))
    assert log_loss(y, p) == pytest.approx(expected)


def test_ece_small_for_calibrated_large_for_overconfident():
    y, p = _calibrated(20_000)
    over = np.clip(1 / (1 + np.exp(-3 * np.log(p / (1 - p + 1e-12) + 1e-12))), 0, 1)
    assert ece(y, p) < 0.02
    assert ece(y, over) > 0.08


def test_reliability_table_counts_sum_to_n():
    y, p = _calibrated(500, seed=3)
    t = reliability_table(y, p, n_bins=10)
    assert t["n"].sum() == 500
    assert (t["mean_p"].between(t["lo"], t["hi"])).all()


def test_murphy_decomposition_exact_for_discrete_forecasts():
    rng = np.random.default_rng(1)
    p = rng.choice([0.05, 0.25, 0.55, 0.85], size=2_000)
    y = (rng.uniform(size=2_000) < p).astype(float)
    d = murphy_decomposition(y, p, n_bins=10)
    assert d["resid"] == pytest.approx(0.0, abs=1e-12)
    assert brier(y, p) == pytest.approx(d["rel"] - d["res"] + d["unc"])


def test_spiegelhalter_rejects_overconfidence_only():
    y, p = _calibrated(5_000, seed=2)
    _, p_cal = spiegelhalter(y, p)
    over = np.where(p > 0.5, 0.97, 0.03)
    _, p_over = spiegelhalter(y, over)
    assert p_cal > 0.01
    assert p_over < 1e-6


def test_auroc_matches_definition_with_ties():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.5, 0.5, 0.8, 0.9])
    # pairs (pos, neg): 6; wins: 0.5 vs .1 win, .5 vs .5 tie, .8/.9 beat both -> 1 + .5 + 4 = 5.5
    assert auroc(y, p) == pytest.approx(5.5 / 6)


def test_rps_and_multiclass_brier():
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    y = np.array([0, 2, 1])
    rps = rps_per(y, P)
    assert rps[0] == 0
    # truth level 2, all mass on level 1: cdf diff = [0, 1] -> 1 / (K - 1) = 0.5
    assert rps[1] == pytest.approx(0.5)
    assert rps[2] == pytest.approx(0.25 / 2)
    assert multiclass_brier_per(y, P)[1] == pytest.approx(2.0)


def test_rps_penalizes_distance():
    near = rps_per([2], np.array([[0, 1.0, 0, 0]]))[0]
    far = rps_per([2], np.array([[1.0, 0, 0, 0]]))[0]
    assert near < far


def test_top_label_ece_zero_when_confidence_matches_hit_rate():
    P = np.tile([0.5, 0.3, 0.2], (1000, 1))
    y = np.array([0] * 500 + [1] * 500)
    assert top_label_ece(y, P) == pytest.approx(0.0)


def test_newey_west_close_to_sample_variance_for_iid():
    d = np.random.default_rng(4).normal(size=5_000)
    assert newey_west_var(d) == pytest.approx(np.var(d), rel=0.1)


def test_compare_detects_better_forecaster():
    rng = np.random.default_rng(5)
    y, p = _calibrated(400, seed=6)
    noisy = np.clip(p + rng.normal(0, 0.25, 400), 0, 1)
    res = compare_losses((p - y) ** 2, (noisy - y) ** 2, n_boot=500, seed=1)
    assert res["diff"] < 0
    assert res["ci_hi"] < 0
    assert res["dm_p"] < 0.05


def test_compare_handles_identical_losses():
    loss = np.full(50, 0.1)
    res = compare_losses(loss, loss, n_boot=100)
    assert res["diff"] == 0
    assert np.isnan(res["dm_stat"])


def test_block_bootstrap_ci_covers_mean():
    d = np.random.default_rng(7).normal(0.3, 1, 300)
    lo, hi = block_bootstrap_ci(d, n_boot=1000, seed=2)
    assert lo < d.mean() < hi


def test_recalibration_fixes_overconfidence_out_of_time():
    y, p = _calibrated(3_000, seed=8)
    logit = np.log(p / (1 - p + 1e-12) + 1e-12)
    over = 1 / (1 + np.exp(-3 * logit))
    rep = recalibration_report(y, over, order=np.arange(len(y)), n_folds=5)
    assert rep["ece_platt"] < rep["ece_raw"] / 2
