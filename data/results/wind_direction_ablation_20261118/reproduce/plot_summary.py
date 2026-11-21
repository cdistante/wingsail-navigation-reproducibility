"""Plot deterministic route-completion counts and bounded first-gate progress."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_study import METHODS, STUDY, WINDS


LABELS = dict(zip(METHODS, (
    "Standard + grid MPC", "Gamma + grid MPC", "Diffusion + grid MPC",
    "Guarded direction Transformer", "Gamma + LOS NMPC", "Gamma + Direct NMPC",
    "Classical forecast tacks", "Learned 120 s progress", "Learned arrival time",
    "Guarded actuator Transformer",
), strict=True))
SHORT_ROUTES = {
    "brindisi_short": "Brindisi\nshort",
    "brindisi_003": "Brindisi\n001–003",
    "brindisi_004": "Brindisi\n001–004",
    "costiera": "Costiera*",
    "otranto_gate": "Otranto\nfirst gate",
    "otranto_003": "Otranto\n001–003",
    "otranto_005": "Otranto\n001–005",
}


def heatmap(axis, headers, groups, max_n):
    counts = []
    ratios = []
    for method in METHODS:
        rows = [groups[method, header] for header in headers]
        assert all(len(group) == max_n for group in rows)
        counts.append([sum(group) for group in rows])
        ratios.append([sum(group) / max_n for group in rows])
    axis.imshow(ratios, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axis.set_xticks(range(len(headers)))
    axis.set_yticks(range(len(METHODS)))
    for y, row in enumerate(counts):
        for x, count in enumerate(row):
            axis.text(x, y, f"{count}/{max_n}", va="center", ha="center",
                      color="white" if count / max_n > 0.62 else "#102039")
    axis.tick_params(length=0, pad=8)
    for edge in axis.spines.values():
        edge.set_visible(False)


def completion_figure(rows, destination):
    routes, directions = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["scope"] == "full_route":
            completed = row["mission_completed"] == "True"
            routes[row["method"], row["route"]].append(completed)
            directions[row["method"], float(row["base_wind"])].append(completed)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True,
                                       gridspec_kw={"width_ratios": [7, 4]})
    heatmap(left, SHORT_ROUTES, routes, 8)
    heatmap(right, WINDS, directions, 14)
    left.set_yticklabels(LABELS.values())
    left.set_xticklabels(SHORT_ROUTES.values())
    right.set_xticklabels(("N ±5°", "E ±5°", "S ±5°", "W ±5°"))
    left.set_title("Route completion by saved GPX")
    right.set_title("Route completion by wind family")
    fig.text(0.015, 0.025, "* The Costiera WP003 capture circle lies within a blocked GEBCO chart cell.",
             color="#314258", fontsize=9)
    fig.tight_layout(w_pad=2, rect=(0, 0.05, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def progress_figure(rows, destination):
    progress = defaultdict(list)
    for row in rows:
        if row["scope"] == "bounded_progress":
            gain = float(row["first_target_range_initial_m"]) - float(row["first_target_closest_m"])
            progress[row["method"], row["route"]].append(gain / 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for axis, route, title in zip(axes, ("naples", "corfu"), ("Naples", "North Corfu"), strict=True):
        for index, method in enumerate(METHODS):
            values = progress[method, route]
            assert len(values) == 8
            axis.scatter(values, np.full(len(values), index), c="#1876a1", s=20, alpha=0.42,
                         label="Individual wind case" if index == 0 else None)
            axis.scatter([np.median(values)], [index], color="#f07443", marker="D", s=28, zorder=3,
                         label="Median of eight cases" if index == 0 else None)
        axis.set_title(title)
        axis.set_xlabel("Best reduction in range to first waypoint [km]")
        axis.grid(axis="x", color="#e2e7ea")
        for edge in ("left", "right", "top"):
            axis.spines[edge].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=8)
    axes[0].set_yticks(range(len(METHODS)), LABELS.values())
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2, rect=(0, 0.07, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    study = parser.parse_args().study
    with (study / "run_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    completion_figure(rows, study / "completion_figure.pdf")
    progress_figure(rows, study / "bounded_progress_figure.pdf")


if __name__ == "__main__":
    main()
