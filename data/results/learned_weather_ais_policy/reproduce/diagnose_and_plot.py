"""Regenerate the held-out comparison figure and first-trim counterfactuals."""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_joint import STUDY, branch_outcomes, episode_keys, init_workers, inputs, setup
from environment_joint import evaluate


OTRANTO = "otranto_short_test_loop__original__leg02__wind_"
METHODS = ("physics", "explicit", "weather", "ais", "joint")
LABELS = ("Physics", "Explicit only", "Weather embedding", "AIS inputs", "Weather + AIS")


def load_episodes():
    outcomes = defaultdict(dict)
    with (STUDY / "test_episodes.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            outcomes[(row["scenario"], row["traffic"])][row["method"]] = row
    failures = {key: methods for key, methods in outcomes.items()
                if methods["physics"]["status"] == "captured" and methods["joint"]["status"] != "captured"}
    tracks = defaultdict(list)
    with gzip.open(STUDY / "test_leg_trajectories.csv.gz", "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["scenario"], row["profile"]
            if key in failures and row["method"] in ("physics", "joint"):
                tracks[key, row["method"]].append(row)
    return outcomes, failures, tracks


def diagnose(failures, tracks):
    import train_joint

    split, archive = inputs()
    keys = {(key.scenario, key.traffic): key for key in episode_keys(split["test"], archive)}
    init_workers()
    observations = []
    for case_id, legs in sorted(failures.items()):
        key = keys[case_id]
        case = train_joint.CASE_INDEX[key.scenario]
        context = setup(key)
        first = evaluate(context, 0., case.target[None], np.array([case.bow]))
        actions = [int(tracks[(case_id, method)][0]["selected_action"]) for method in ("physics", "joint")]
        if not all(first.allowed[0, actions]):
            raise ValueError("A logged first trim was outside the common report mask.")
        continuations = branch_outcomes(context, first, np.array(actions), 0.)
        row = {"scenario": key.scenario, "traffic": key.traffic, "initial_bow_deg": case.bow,
               "wind_from_deg": case.first.polar.wind_from, "wind_mps": case.first.wind_speed,
               "current_to_en_mps": case.first.polar.flow.tolist(), "mark_en_m": case.target.tolist(),
               "data": {}}
        for j, method in enumerate(("physics", "joint")):
            frames = tracks[case_id, method]
            nearest_fix = min(np.linalg.norm(case.target - np.array([float(frame["end_east"]),
                                                                    float(frame["end_north"])]))
                              for frame in frames)
            row["data"][method] = {
                "first_candidate_index": actions[j],
                "first_end_bow_deg": float(frames[0]["nominal_end_bow"]),
                "first_applied_report_margin_m": float(frames[0]["report_margin_m"]),
                "recorded_time_s": float(legs[method]["duration_s"]),
                "recorded_outcome": legs[method]["status"],
                "closest_logged_fix_m": float(nearest_fix),
                "first_trim_then_physics": {
                    "status": "captured" if continuations[j] <= train_joint.MAX_SECONDS else "uncaptured",
                    "time_s": float(continuations[j]) if continuations[j] <= train_joint.MAX_SECONDS else None,
                },
            }
        observations.append(row)
    (STUDY / "first_trim_diagnostic.json").write_text(json.dumps(observations, indent=2) + "\n")
    return observations


def clip_track(rows, time_limit_s):
    clock = [float(rows[0]["elapsed_s"])]
    positions = [(float(rows[0]["start_east"]), float(rows[0]["start_north"]))]
    for row in rows:
        t = float(row["elapsed_s"]) + float(row["applied_s"])
        if t > time_limit_s:
            break
        clock.append(t)
        positions.append((float(row["end_east"]), float(row["end_north"])))
    return np.asarray(clock), np.asarray(positions)


def plot(outcomes, tracks, diagnostics):
    result = json.loads((STUDY / "test_summary.json").read_text())
    totals = result["strata"]["all"]
    colors = ["#315069", "#96A4AD", "#96A4AD", "#96A4AD", "#0C756E"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw={"width_ratios": [1., 1.15]})
    capture, timing, safety, approach = axes.flat
    indices = np.arange(len(METHODS))
    amounts = [totals[method]["captured"] for method in METHODS]
    capture.barh(indices, amounts, color=colors, height=.56)
    capture.set(yticks=indices, yticklabels=LABELS, xlim=(0, 216), xlabel="Captured waypoint cells / 216",
                title="Completion on held-out legs")
    capture.invert_yaxis()
    for y, count in zip(indices, amounts):
        capture.text(count+4, y, str(count), va="center", fontweight="medium")

    for y, method in enumerate(METHODS[1:], start=1):
        gaps = [float(group[method]["duration_s"])-float(group["physics"]["duration_s"])
                for group in outcomes.values() if group[method]["status"] == group["physics"]["status"] == "captured"]
        low, mid, high = np.quantile(gaps, [.1, .5, .9])/60.
        timing.plot([low, high], [y, y], color=colors[y], linewidth=3, solid_capstyle="round")
        timing.scatter([mid], [y], color=colors[y], s=50, zorder=3, edgecolor="white", linewidth=.6)
    timing.axvline(0, color="#697984", linestyle="--", linewidth=.75)
    timing.set(yticks=indices[1:], yticklabels=LABELS[1:],
               xlabel="Time difference versus physics (min)",
               title="Paired capture times · median and 10th–90th percentiles")
    timing.invert_yaxis()

    margins = [totals[method]["min_applied_truth_margin_m"] for method in METHODS]
    safety.barh(indices, margins, color=colors, height=.56)
    safety.set(yticks=indices, yticklabels=LABELS, xlim=(0, max(margins)*1.28),
               xlabel="Minimum swept truth margin beyond buffer and reserve (m)",
               title="Full AIS clearance · all applied steps")
    safety.invert_yaxis()
    for y, value in zip(indices, margins, strict=True):
        safety.text(value+.008, y, f"{value:.3f}", va="center", fontweight="medium")

    example = next(row for row in diagnostics if row["scenario"] == OTRANTO+"005"
                   and row["traffic"] == "mixed_sharp_dropout")
    key = example["scenario"], example["traffic"]
    mark = np.asarray(example["mark_en_m"])
    for method, color, label in (("physics", colors[0], LABELS[0]), ("joint", colors[-1], LABELS[-1])):
        time, points = clip_track(tracks[key, method], 3600.)
        ranges = np.linalg.norm(points-mark, axis=1)
        approach.plot(time/60., ranges, color=color, linewidth=1.9, label=label)
        approach.scatter(time[-1]/60., ranges[-1], color=color, s=22, zorder=3)
    approach.axhline(50., color="#697984", linestyle="--", linewidth=.8)
    approach.text(59., 57., "50 m capture radius", color="#596A73", ha="right", fontsize=8)
    approach.set(xlim=(0., 61.), ylim=(0., None), xlabel="Minutes since start on leg 002–003",
                 ylabel="Range to 003 (m)", title="Otranto example · wind from 5° · sharp AIS dropout")
    approach.legend(loc="upper right", frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.grid(axis="x", linewidth=.6, alpha=.30)
        axis.set_axisbelow(True)
    fig.suptitle("Transformer weather and AIS policy on held-out legs", fontsize=17, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .95), w_pad=2.8, h_pad=2.4)
    fig.savefig(STUDY / "heldout_comparison.png", dpi=205, bbox_inches="tight")
    plt.close(fig)


def main():
    outcomes, failures, tracks = load_episodes()
    details = diagnose(failures, tracks)
    plot(outcomes, tracks, details)


if __name__ == "__main__":
    main()
