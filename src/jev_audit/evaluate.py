"""Join cached predictions with ground truth and compute every table in the report.

All metric tables use the COMMON sample: (state, question) items that every enabled arm
answered without error. Coverage per arm is reported separately in the ops table, so a model
that fails on hard items cannot look better by skipping them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jev_audit.config import ExperimentConfig
from jev_audit.data.states import read_states
from jev_audit.metrics.binary import binary_summary, brier_per, reliability_table
from jev_audit.metrics.compare import compare_losses
from jev_audit.metrics.multiclass import multiclass_brier_per, multiclass_summary, rps_per
from jev_audit.metrics.recalibrate import recalibration_report
from jev_audit.questions import QuestionBank
from jev_audit.runner import load_labels, predictions_path


@dataclass
class EvalTables:
    binary: pd.DataFrame
    multiclass: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    recal: pd.DataFrame
    ops: pd.DataFrame
    reliability: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def save(self, out_dir) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("binary", "multiclass", "summary", "pairwise", "recal", "ops", "reliability"):
            getattr(self, name).to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))


# ---------------------------------------------------------------------------- loading
def load_predictions(cfg: ExperimentConfig, bank: QuestionBank) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(answers, calls): answers = one row per (arm, state, question) from the latest
    error-free record; calls = one row per API call attempt (for ops / error rates)."""
    fingerprint = bank.fingerprint()
    answers, calls = [], []
    for arm in cfg.enabled_arms():
        path = predictions_path(cfg, arm)
        if not path.exists():
            continue
        latest: dict[str, dict] = {}
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("question_set") != fingerprint:
                    continue
                calls.append(
                    {
                        "arm": arm,
                        "state_id": rec["state_id"],
                        "model": rec.get("model"),
                        "ok": not rec.get("error"),
                        "latency_s": rec.get("latency_s"),
                        "input_tokens": rec.get("input_tokens"),
                        "output_tokens": rec.get("output_tokens"),
                    }
                )
                if not rec.get("error"):
                    latest[rec["state_id"]] = rec
        for rec in latest.values():
            for qid, ans in rec["answers"].items():
                if qid not in bank.questions:
                    continue
                answers.append(
                    {
                        "arm": arm,
                        "state_id": rec["state_id"],
                        "week": rec["week"],
                        "view": rec["view"],
                        "qid": qid,
                        "type": ans["type"],
                        "p": ans.get("p"),
                        "probs": ans.get("probs"),
                        "confidence": ans.get("confidence"),
                        "abstained": ans.get("abstained"),
                    }
                )
    return pd.DataFrame(answers), pd.DataFrame(calls)


def _item_loss(group: pd.DataFrame, qtype: str) -> np.ndarray:
    if qtype == "noul":
        return brier_per(group["y"], group["p"])
    probs = np.vstack(group["probs"].to_numpy())
    if qtype == "score":
        return rps_per(group["y"], probs)
    return multiclass_brier_per(group["y"], probs)


LOSS_NAME = {"noul": "brier", "choice": "mbrier", "score": "rps"}


# ---------------------------------------------------------------------------- tables
def evaluate(cfg: ExperimentConfig) -> EvalTables:
    bank = QuestionBank.from_yaml(cfg.questions)
    ev = cfg.evaluation
    labels = load_labels(cfg)
    answers, calls = load_predictions(cfg, bank)
    if answers.empty:
        raise RuntimeError(f"No predictions found under {cfg.raw_predictions_dir}. Run arms first.")

    df = answers.merge(labels[["week", "qid", "horizon", "y"]], on=["week", "qid"], how="inner")
    arms = sorted(df["arm"].unique())
    counts = df.groupby(["state_id", "qid"])["arm"].nunique()
    common = counts[counts == len(arms)].index
    df = df.set_index(["state_id", "qid"]).loc[common].reset_index()
    df = df.sort_values(["arm", "view", "qid", "week"]).reset_index(drop=True)

    # ---- binary (noul) metrics
    rows, rel_rows = [], []
    for (arm, view, qid), g in df[df["type"] == "noul"].groupby(["arm", "view", "qid"]):
        stats = binary_summary(g["y"], g["p"], ev.n_bins)
        rows.append({"arm": arm, "view": view, "qid": qid, "horizon": bank[qid].horizon, **stats})
        for r in reliability_table(g["y"], g["p"], ev.n_bins).itertuples(index=False):
            rel_rows.append(
                {"arm": arm, "view": view, "qid": qid, "horizon": bank[qid].horizon, **r._asdict()}
            )
    # pooled reliability per (arm, view, horizon) for the headline plot
    for (arm, view, h), g in df[df["type"] == "noul"].groupby(["arm", "view", "horizon"]):
        for r in reliability_table(g["y"], g["p"], ev.n_bins).itertuples(index=False):
            rel_rows.append({"arm": arm, "view": view, "qid": "ALL_NOUL", "horizon": h, **r._asdict()})
    binary = pd.DataFrame(rows)
    if ev.reference_arm and not binary.empty and ev.reference_arm in arms:
        ref = binary[binary["arm"] == ev.reference_arm][["view", "qid", "brier"]]
        binary = binary.merge(ref.rename(columns={"brier": "brier_ref"}), on=["view", "qid"], how="left")
        binary["bss"] = 1.0 - binary["brier"] / binary["brier_ref"]
        binary = binary.drop(columns="brier_ref")

    # ---- choice / score metrics
    rows = []
    for (arm, view, qid), g in df[df["type"] != "noul"].groupby(["arm", "view", "qid"]):
        q = bank[qid]
        conf = g["confidence"]
        confidence = conf.to_numpy(dtype=float) if conf.notna().all() else None
        stats = multiclass_summary(
            g["y"],
            np.vstack(g["probs"].to_numpy()),
            ordinal=q.type == "score",
            n_bins=ev.n_bins,
            confidence=confidence,
        )
        rows.append({"arm": arm, "view": view, "qid": qid, "type": q.type, "horizon": q.horizon, **stats})
    multiclass = pd.DataFrame(rows)

    # ---- headline summary per (arm, view, horizon)
    summary = _summary(binary, multiclass, df, ev.n_bins)

    # ---- paired comparisons
    pairs: list[tuple[str, str]] = []
    if ev.primary_pair and all(a in arms for a in ev.primary_pair):
        pairs.append(tuple(ev.primary_pair))
    if ev.reference_arm in arms:
        pairs += [
            (a, ev.reference_arm)
            for a in arms
            if a != ev.reference_arm and (a, ev.reference_arm) not in pairs
        ]
    pairwise = _pairwise(df, bank, pairs, ev.bootstrap_n, ev.bootstrap_seed)

    # ---- recalibration (noul only)
    rows = []
    for (arm, view, qid), g in df[df["type"] == "noul"].groupby(["arm", "view", "qid"]):
        order = pd.to_datetime(g["week"]).map(pd.Timestamp.toordinal).to_numpy()
        rows.append(
            {
                "arm": arm,
                "view": view,
                "qid": qid,
                "horizon": bank[qid].horizon,
                **recalibration_report(g["y"], g["p"], order, ev.recal_folds, ev.n_bins),
            }
        )
    recal = pd.DataFrame(rows)

    # ---- ops: coverage, errors, latency, tokens, cost
    n_states = len(read_states(cfg.data.states_path))
    ops = _ops(cfg, calls, n_states)

    meta = {
        "run_name": cfg.run_name,
        "data_source": cfg.data.source,
        "question_set": bank.fingerprint(),
        "arms": arms,
        "n_states": n_states,
        "n_common_items": int(len(common)),
        "weeks": [str(df["week"].min()), str(df["week"].max())],
        "views": sorted(df["view"].unique().tolist()),
        "synthetic": cfg.data.source == "synthetic" or any(cfg.arms[a].kind == "mock" for a in arms),
    }
    return EvalTables(binary, multiclass, summary, pairwise, recal, ops, pd.DataFrame(rel_rows), meta)


def _summary(binary: pd.DataFrame, multiclass: pd.DataFrame, df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    from jev_audit.metrics.binary import ece

    parts = []
    if not binary.empty:
        b = binary.groupby(["arm", "view", "horizon"]).agg(
            noul_brier=("brier", "mean"),
            noul_log_loss=("log_loss", "mean"),
            noul_ece=("ece", "mean"),
            noul_accuracy=("accuracy", "mean"),
        )
        pooled = (
            df[df["type"] == "noul"]
            .groupby(["arm", "view", "horizon"])
            .apply(lambda g: ece(g["y"], g["p"], n_bins), include_groups=False)
            .rename("noul_ece_pooled")
        )
        parts += [b, pooled]
    if not multiclass.empty:
        sc = multiclass[multiclass["type"] == "score"].groupby(["arm", "view", "horizon"])["rps"].mean()
        ch = multiclass[multiclass["type"] == "choice"].groupby(["arm", "view", "horizon"])["mbrier"].mean()
        parts += [sc.rename("score_rps"), ch.rename("choice_mbrier")]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).reset_index()


def _pairwise(df, bank, pairs, n_boot, seed) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        da, db = df[df["arm"] == a], df[df["arm"] == b]
        for (view, qid), ga in da.groupby(["view", "qid"]):
            gb = db[(db["view"] == view) & (db["qid"] == qid)]
            m = ga.merge(gb, on=["state_id", "week", "y"], suffixes=("_a", "_b")).sort_values("week")
            if len(m) < 3:
                continue
            q = bank[qid]
            la = _item_loss(m.rename(columns={"p_a": "p", "probs_a": "probs"}), q.type)
            lb = _item_loss(m.rename(columns={"p_b": "p", "probs_b": "probs"}), q.type)
            rows.append(
                {
                    "arm_a": a,
                    "arm_b": b,
                    "view": view,
                    "qid": qid,
                    "horizon": q.horizon,
                    "loss": LOSS_NAME[q.type],
                    **compare_losses(la, lb, n_boot, seed),
                }
            )
        # pooled over noul questions: one mean-Brier value per state
        for (view, h), ga in da[da["type"] == "noul"].groupby(["view", "horizon"]):
            gb = db[(db["view"] == view) & (db["horizon"] == h) & (db["type"] == "noul")]
            m = ga.merge(gb, on=["state_id", "week", "qid", "y"], suffixes=("_a", "_b"))
            if m.empty:
                continue
            m["la"] = (m["p_a"] - m["y"]) ** 2
            m["lb"] = (m["p_b"] - m["y"]) ** 2
            per_state = m.groupby(["state_id", "week"])[["la", "lb"]].mean().reset_index().sort_values("week")
            if len(per_state) < 3:
                continue
            rows.append(
                {
                    "arm_a": a,
                    "arm_b": b,
                    "view": view,
                    "qid": "ALL_NOUL",
                    "horizon": h,
                    "loss": "brier",
                    **compare_losses(per_state["la"], per_state["lb"], n_boot, seed),
                }
            )
    return pd.DataFrame(rows)


def _ops(cfg: ExperimentConfig, calls: pd.DataFrame, n_states: int) -> pd.DataFrame:
    rows = []
    if calls.empty:
        return pd.DataFrame()
    for arm, g in calls.groupby("arm"):
        ok = g[g["ok"]]
        arm_cfg = cfg.arms[arm]
        lat = ok["latency_s"].dropna()
        tin = ok["input_tokens"].dropna()
        tout = ok["output_tokens"].dropna()
        price = arm_cfg.price_per_mtok or {}
        cost = None
        if arm_cfg.kind in ("jev", "llm") and price.get("input") is not None and len(tin):
            mean_out = float(tout.mean()) if len(tout) else 0.0
            per_call = (float(tin.mean()) * price["input"] + mean_out * (price.get("output") or 0.0)) / 1e6
            cost = per_call * 1000
        rows.append(
            {
                "arm": arm,
                "kind": arm_cfg.kind,
                "model": ok["model"].dropna().iloc[-1] if ok["model"].notna().any() else arm_cfg.model,
                "calls": int(len(g)),
                "errors": int((~g["ok"]).sum()),
                "coverage": ok["state_id"].nunique() / n_states if n_states else float("nan"),
                "latency_median_s": float(lat.median()) if len(lat) else None,
                "latency_p95_s": float(lat.quantile(0.95)) if len(lat) else None,
                "input_tokens_mean": float(tin.mean()) if len(tin) else None,
                "output_tokens_mean": float(tout.mean()) if len(tout) else None,
                "usd_per_1k_states": cost,
            }
        )
    return pd.DataFrame(rows)
