"""Charts from results/summary_by_strategy.csv: python -m ptr.charts [results]."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .frameworks import STEADY_MODE  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated, adjacent pairs, light mode
MARKERS = ["o", "s", "^", "D"]  # second encoding next to colour
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# (strategy, plan cache mode, legend label); slot 1 is always the untuned baseline
CHARTS = {
    "recall-prepared.png": (
        "Tenant filter on a column, prepared statements (generic plan)",
        [
            ("hnsw_default", "generic", "HNSW, defaults"),
            ("iter_relaxed", "generic", "iterative_scan relaxed_order"),
            ("iter_relaxed_100k", "generic", "relaxed_order + max_scan_tuples 100k"),
            ("btree_planner", "generic", "B-tree on tenant_id"),
        ],
    ),
    "recall-jsonb.png": (
        "Tenant filter inside JSONB metadata, plain queries (custom plan)",
        [
            ("jsonb_default", "custom", "metadata->>'user_id', HNSW defaults"),
            ("jsonb_relaxed", "custom", "same + relaxed_order"),
            ("jsonb_expr_index", "custom", "same + expression B-tree"),
            ("hnsw_default", "custom", "tenant_id column, HNSW defaults"),
        ],
    ),
}


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_lines(rows: list[dict], out: Path, title: str, series: list[tuple[str, str, str]]) -> None:
    builds = max(int(r["builds"]) for r in rows)
    buckets = sorted({int(r["bucket"]) for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    x = list(range(len(buckets)))
    for i, (name, mode, label) in enumerate(series):
        pts = {int(r["bucket"]): r for r in rows if r["strategy"] == name and r["plan_cache"] == mode}
        if not pts:
            continue
        y = [float(pts[b]["mean_recall"]) for b in buckets]
        lo = [y[j] - float(pts[b]["recall_low"]) for j, b in enumerate(buckets)]
        hi = [float(pts[b]["recall_high"]) - y[j] for j, b in enumerate(buckets)]
        # small horizontal nudge so overlapping series at 1.0 stay visible
        xs = [xi + (i - 1.5) * 0.06 for xi in x]
        ax.errorbar(
            xs,
            y,
            yerr=[lo, hi],
            color=SERIES[i],
            linewidth=2,
            marker=MARKERS[i],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            elinewidth=1.2,
            capsize=0,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, [f"{b:,} row{'s' if b > 1 else ''}" for b in buckets])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("tenant size (rows the tenant owns in a 200k-row table)", color=TEXT_2, fontsize=10)
    ax.set_ylabel("recall@10 vs exact search", color=TEXT_2, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12.5, loc="left", pad=14)
    ax.text(
        0,
        -0.17,
        f"Mean over {builds} HNSW builds, whiskers = lowest and highest build. 60 queries per point.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def recall_heatmap(rows: list[dict], out: Path, title: str, panels: dict[str, str] | None = None) -> None:
    """One panel per plan cache mode; with `panels` ({strategy: mode}) a single panel of chosen cells."""
    buckets = sorted({int(r["bucket"]) for r in rows})
    names = list(dict.fromkeys(r["strategy"] for r in rows))
    labels = {r["strategy"]: r["label"] for r in rows}
    if panels:
        rows = [{**r, "plan_cache": "steady"} for r in rows if panels.get(r["strategy"]) == r["plan_cache"]]
        labels = {n: f"{labels[n]}{'  [prepared]' if panels[n] == 'generic' else ''}" for n in names}
    modes = [m for m in ("custom", "generic", "steady") if any(r["plan_cache"] == m for r in rows)]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(5.2 * len(modes) + 3, 0.42 * len(names) + 1.8),
        dpi=150,
        facecolor=SURFACE,
        sharey=True,
        squeeze=False,
    )
    for ax, mode in zip(axes[0], modes, strict=True):
        cell = {(r["strategy"], int(r["bucket"])): float(r["mean_recall"]) for r in rows if r["plan_cache"] == mode}
        grid = [[cell.get((n, b), float("nan")) for b in buckets] for n in names]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for yi, n in enumerate(names):
            for xi, b in enumerate(buckets):
                v = cell.get((n, b))
                if v is None:
                    continue
                ax.text(
                    xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if v >= 0.55 else TEXT
                )
        # 2px surface gaps between cells
        ax.set_xticks([i - 0.5 for i in range(1, len(buckets))], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, len(names))], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0, colors=TEXT_2, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(len(buckets)), [f"{b:,}" for b in buckets])
        ax.set_xlabel("tenant size (rows)", color=TEXT_2, fontsize=9)
        panel = {
            "custom": "custom plan per query",
            "generic": "generic plan (prepared statement)",
            "steady": "the plan each store's driver settles on",
        }[mode]
        ax.set_title(panel, color=TEXT, fontsize=11, loc="left")
    axes[0][0].set_yticks(range(len(names)), [labels[n] for n in names])
    fig.suptitle(title, color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    folder = Path((argv or sys.argv[1:] or ["results"])[0])
    rows = load(folder / "summary_by_strategy.csv")
    present = {r["strategy"] for r in rows}
    for name, (title, series) in CHARTS.items():
        if all(n in present for n, _, _ in series):
            recall_lines(rows, folder / name, title, series)
    fw = {r["strategy"] for r in rows if r["strategy"] in STEADY_MODE}
    base = [r for r in rows if r["strategy"] not in fw]
    builds = max(int(r["builds"]) for r in rows)
    note = f"(mean of {builds} HNSW build{'s' if builds > 1 else ''})"
    if base:
        recall_heatmap(base, folder / "recall-heatmap.png", f"Mean recall@10 by strategy and tenant size {note}")
    if fw:
        recall_heatmap(
            [r for r in rows if r["strategy"] in fw],
            folder / "recall-frameworks.png",
            f"Framework query shapes, recall@10 by tenant size {note}",
            panels={n: STEADY_MODE[n] for n in fw},
        )
    print(f"wrote charts to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
