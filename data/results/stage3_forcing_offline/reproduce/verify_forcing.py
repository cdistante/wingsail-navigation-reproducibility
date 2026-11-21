"""Independently check the exported records and rerun pinned forcing forecasts."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta
import gzip
from hashlib import file_digest
import json
from math import hypot, isclose, sqrt
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("wind_e_mps", "wind_n_mps", "current_e_mps", "current_n_mps", "wave_height_m")
METHODS = ("persistence", "causal_trend", "transformer")
HOUR = timedelta(hours=1)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def vector(row, prefix):
    return tuple(float(row[f"{prefix}_{key}"]) for key in CHANNELS)


def check_records(root):
    selection = json.loads((root / "selection_receipt.json").read_text())
    settings = selection["settings"]
    report = json.loads((root / "metrics.json").read_text())
    note = {"counts_by_block": {}, "checks": []}
    assert settings["trend_multiplier"] == min(settings["validation_trend_candidates"],
                                                key=lambda candidate: (candidate["score"], candidate["alpha"]))["alpha"]
    assert settings["ridge"] == min(settings["validation_ridge_candidates"],
                                     key=lambda candidate: (candidate["score"], candidate["penalty"]))["penalty"]
    assert set(selection["pretest_blocks"]) == {"winter", "spring", "summer"}
    note["checks"].append("Selected hyperparameters minimize the exported April objective within each model family.")
    with gzip.open(root / "prediction_records.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(row["block"], row["leg"], datetime.fromisoformat(row["issue_utc"])): row for row in rows}
    assert len(keys) == len(rows)
    counts = Counter()
    sum_squared_error = Counter()
    matching_previous = matching_next = 0
    caps = settings["change_caps"]
    rate = settings["trend_multiplier"]
    for (block, leg, issue), row in keys.items():
        assert datetime.fromisoformat(row["target_utc"]) == issue + HOUR
        observed, past, target = vector(row, "issue"), vector(row, "previous_hour"), vector(row, "target")
        assert vector(row, "persistence") == observed
        previous = keys.get((block, leg, issue - HOUR))
        following = keys.get((block, leg, issue + HOUR))
        if previous:
            assert past == vector(previous, "issue")
            matching_previous += 1
        if following:
            assert target == vector(following, "issue")
            matching_next += 1
        trend_delta = [rate * (current - old) for current, old in zip(observed, past)]
        for start, maximum in zip((0, 2), caps):
            length = hypot(*trend_delta[start:start + 2])
            if length > maximum:
                trend_delta[start:start + 2] = [component * maximum / length for component in trend_delta[start:start + 2]]
        trend_delta[-1] = max(-caps[-1], min(caps[-1], trend_delta[-1]))
        projected = [current + delta for current, delta in zip(observed, trend_delta)]
        projected[-1] = max(0., projected[-1])
        assert all(isclose(saved, independent, abs_tol=1e-12) for saved, independent in zip(vector(row, "causal_trend"), projected))
        counts[block] += 1
        for method in METHODS:
            forecast = vector(row, method)
            sum_squared_error[block, method, "wind"] += (forecast[0] - target[0])**2 + (forecast[1] - target[1])**2
            sum_squared_error[block, method, "current"] += (forecast[2] - target[2])**2 + (forecast[3] - target[3])**2

    for block, count in counts.items():
        assert count == report["leg_count"] * 168 == report["blocks"][block]["archive"]["count"]
        for method in METHODS:
            metrics = report["blocks"][block]["metrics"]["all"][method]
            for field in ("wind", "current"):
                independent = sqrt(sum_squared_error[block, method, field] / count)
                assert isclose(metrics[f"{field}_en_vector_rmse_mps"], independent, abs_tol=1e-12)
    note["counts_by_block"] = dict(counts)
    note["previous_and_next_hour_pair_checks"] = [matching_previous, matching_next]
    note["checks"].extend(("Unique paired rows and exact one-hour forecast horizons.",
                           "Previous and next weather vectors match corresponding adjacent issue records.",
                           "Independent trend forecasts and vector RMSE reproduce saved values."))
    return note


def main():
    note = check_records(ROOT)
    files = ("selection_receipt.json", "forcing_ridge_head.npz", "prediction_records.csv.gz", "daily_metrics.csv", "metrics.json")
    original = {name: digest(ROOT / name) for name in files}
    with TemporaryDirectory(prefix="stage3_forcing_verify_") as folder:
        subprocess.run([sys.executable, "-B", str(ROOT / "reproduce" / "run_forcing.py"), "--output", folder], check=True)
        reproduced = {name: digest(Path(folder) / name) for name in files}
    assert reproduced == original
    note["checks"].append("Fresh independent process reproduced all five study artifacts byte for byte.")
    note["artifact_sha256"] = original
    (ROOT / "verification.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {sum(note['counts_by_block'].values())} paired forecasts and byte-identical reproduction.")


if __name__ == "__main__":
    main()
