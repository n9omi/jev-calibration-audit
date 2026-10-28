"""End-to-end: the offline demo config runs every stage and the numbers behave as designed."""

import pandas as pd

from jev_audit import pipeline
from jev_audit.config import load_config
from jev_audit.runner import run_arm


def test_demo_pipeline_end_to_end(repo_root, tmp_path):
    cfg = load_config(repo_root / "configs" / "demo.yaml", out_root=tmp_path)
    logs = []
    pipeline.fetch(cfg, log=logs.append)
    pipeline.build(cfg, log=logs.append)
    pipeline.run_enabled(cfg, log=logs.append)
    tables = pipeline.evaluate_and_report(cfg, log=logs.append)

    report = (cfg.report_dir / "report.md").read_text()
    assert "SYNTHETIC DEMO" in report
    assert (cfg.report_dir / "figures" / "reliability.png").exists()
    assert (cfg.results_dir / "binary.csv").exists()

    s = tables.summary.set_index(["arm", "view", "horizon"])
    # rules parse templated text perfectly on this-week questions
    assert (s.xs(0, level="horizon").loc["rules", "noul_accuracy"] == 1.0).all()
    # the overconfident mock is worse calibrated on forecasts than the calibrated one
    h1 = s.xs(1, level="horizon")
    assert (
        h1.loc["mock_overconfident", "noul_ece_pooled"] > h1.loc["mock_calibrated", "noul_ece_pooled"]
    ).all()
    # rules fall back to climatology on forecasts
    assert (h1.loc["rules", "noul_brier"] - h1.loc["climatology", "noul_brier"]).abs().max() < 1e-12

    # rerunning an arm is a no-op thanks to the cache
    path = run_arm(cfg, "rules", log=logs.append)
    assert sum(1 for _ in path.open()) == len(pd.read_json(cfg.data.states_path, lines=True))
    assert any("to run" in line and " 0 to run" in line for line in logs)
