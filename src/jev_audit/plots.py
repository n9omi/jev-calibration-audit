"""Static figures for the markdown report (PNG, light surface).

Colors: validated categorical slots (blue, orange, aqua, yellow), assigned to arms in a fixed
order (primary pair first, then config order) so an arm keeps its color across figures and
runs. Every series also gets its own marker shape, and the report carries the same numbers
as tables, so no reading depends on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
REF = "#9a9893"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def arm_styles(arms: list[str]) -> dict[str, tuple[str, str]]:
    return {a: (SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]) for i, a in enumerate(arms)}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def reliability_figure(rel: pd.DataFrame, arms: list[str], path: Path, title: str) -> Path | None:
    """Pooled reliability diagrams: rows = view, columns = horizon (all Noul questions)."""
    pooled = rel[rel["qid"] == "ALL_NOUL"]
    if pooled.empty:
        return None
    views = sorted(pooled["view"].unique())
    horizons = sorted(pooled["horizon"].unique())
    styles = arm_styles(arms)
    fig, axes = plt.subplots(
        len(views),
        len(horizons),
        figsize=(4.2 * len(horizons), 4.0 * len(views)),
        squeeze=False,
        facecolor=SURFACE,
    )
    for r, view in enumerate(views):
        for c, h in enumerate(horizons):
            ax = axes[r][c]
            _style_axes(ax)
            ax.plot([0, 1], [0, 1], color=REF, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            sub = pooled[(pooled["view"] == view) & (pooled["horizon"] == h)]
            for arm in arms:
                g = sub[sub["arm"] == arm].sort_values("mean_p")
                if g.empty:
                    continue
                color, marker = styles[arm]
                ax.plot(g["mean_p"], g["freq"], color=color, linewidth=2, zorder=2, label=arm)
                ax.scatter(
                    g["mean_p"],
                    g["freq"],
                    s=14 + 70 * g["n"] / g["n"].max(),
                    color=color,
                    marker=marker,
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    zorder=3,
                )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            horizon_name = "this week (h=0)" if h == 0 else f"forecast (h={h})"
            ax.set_title(f"{view} · {horizon_name}", fontsize=9, color=TEXT, loc="left")
            if r == len(views) - 1:
                ax.set_xlabel("Forecast probability (bin mean)", fontsize=8, color=TEXT_2)
            if c == 0:
                ax.set_ylabel("Observed frequency", fontsize=8, color=TEXT_2)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=styles[a][0],
            marker=styles[a][1],
            linewidth=2,
            markersize=6,
            markeredgecolor=SURFACE,
            label=a,
        )
        for a in arms
        if a in set(pooled["arm"])
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.955),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.text(
        0.01,
        0.005,
        "Dashed line = perfect calibration. Marker size ∝ items in bin.",
        fontsize=7,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def pairwise_figure(pw: pd.DataFrame, arm_a: str, arm_b: str, path: Path) -> Path | None:
    """Mean loss difference (A - B) with 95% block-bootstrap CI, one row per question/view."""
    sub = pw[(pw["arm_a"] == arm_a) & (pw["arm_b"] == arm_b)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(["horizon", "qid", "view"]).reset_index(drop=True)
    sub["row"] = (
        "h=" + sub["horizon"].astype(str) + "  " + sub["qid"] + " · " + sub["view"] + " (" + sub["loss"] + ")"
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(sub) + 1.6), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    y = list(range(len(sub)))[::-1]
    ax.axvline(0, color=REF, linewidth=1)
    # separate horizon groups
    for i in range(1, len(sub)):
        if sub.loc[i, "horizon"] != sub.loc[i - 1, "horizon"]:
            ax.axhline(y[i] + 0.5, color=GRID, linewidth=1)
    ax.hlines(y, sub["ci_lo"], sub["ci_hi"], color=SLOTS[0], linewidth=2)
    ax.scatter(sub["diff"], y, color=SLOTS[0], s=28, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["row"], fontsize=7, color=TEXT)
    ax.set_xlabel(f"← {arm_a} better        {arm_b} better →", fontsize=8, color=TEXT_2)
    fig.suptitle(
        f"{arm_a} − {arm_b}: mean paired loss difference (95% block-bootstrap CI)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
