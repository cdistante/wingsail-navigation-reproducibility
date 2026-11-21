"""Fit the published arrival model on the same labels with schedule inputs masked."""

from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from run_wind_scenario import WORKSPACE
from world.learned_tack_arrival import FEATURE_NAMES, LearnedTackArrivalSelector


MASKED_FIELDS = tuple(name for name in FEATURE_NAMES
                      if name in {"has_transition", "first_leg_seconds"} or name.startswith("next_"))
ORIGINAL_STUDY = WORKSPACE / "data/results/learned_tack_arrival_20261118"


def make_masked_dataset(source: Path, destination: Path) -> int:
    with (gzip.open(source, "rt", encoding="utf-8", newline="") as file,
          gzip.open(destination, "wt", encoding="utf-8", newline="") as out):
        reader = csv.DictReader(file)
        if len(MASKED_FIELDS) != 8 or not set(MASKED_FIELDS).issubset(reader.fieldnames):
            raise ValueError("The original training CSV lacks the eight proposed schedule fields.")
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        count = 0
        mask = dict.fromkeys(MASKED_FIELDS, "0")
        for row in reader:
            writer.writerow({**row, **mask})
            count += 1
    return count


def verify_and_mask_deployment(model_file: Path, output_file: Path) -> None:
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    for name in MASKED_FIELDS:
        index = payload["feature_names"].index(name)
        if payload["means"][index] != 0.0 or payload["scales"][index] != 1.0:
            raise ValueError(f"Unexpected normalization of masked field {name!r}.")
        for weights in payload["layers"][0]["weights"]:
            weights[index] = 0.0
    output_file.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    original = LearnedTackArrivalSelector.load(model_file)
    deployed = LearnedTackArrivalSelector.load(output_file)
    rng = np.random.default_rng(20261118)
    unmasked = rng.normal(size=(32, len(FEATURE_NAMES)))
    masked = unmasked.copy()
    masked[:, [FEATURE_NAMES.index(field) for field in MASKED_FIELDS]] = 0.0
    np.testing.assert_array_equal(
        unmasked @ deployed.layers[0][0].T + deployed.layers[0][1],
        masked @ original.layers[0][0].T + original.layers[0][1],
    )


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_data = ORIGINAL_STUDY / "tack_arrival_candidates.csv.gz"
    original_trainer = ORIGINAL_STUDY / "reproduce/train_ranker.py"
    original_model = ORIGINAL_STUDY / "arrival_model.json"
    original_metrics = json.loads((ORIGINAL_STUDY / "offline_metrics.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="arrival_without_transition_") as directory:
        temp = Path(directory)
        masked_file = temp / "candidates.csv.gz"
        rows = make_masked_dataset(original_data, masked_file)
        subprocess.run([sys.executable, "-B", str(original_trainer), "--output", str(temp),
                        "--dataset", str(masked_file), "--epochs", str(original_metrics["epochs"]),
                        "--seed", str(original_metrics["seed"])], cwd=WORKSPACE, check=True)
        model_file = output / "arrival_without_transition.json"
        verify_and_mask_deployment(temp / "arrival_model.json", model_file)
        comparison = json.loads((temp / "offline_metrics.json").read_text(encoding="utf-8"))
    comparison.update({
        "masked_features": MASKED_FIELDS, "label_rows": rows,
        "purpose": "Remove scheduled-continuation inputs while retaining transition-aware Fossen labels.",
        "identical_inputs": {str(path.relative_to(WORKSPACE)): sha256(path.read_bytes()).hexdigest()
                             for path in (original_data, original_trainer, original_model)},
        "deployment_model_sha256": sha256(model_file.read_bytes()).hexdigest(),
        "original_full_model_test_metrics": original_metrics["test"],
    })
    (output / "offline_metrics_without_transition.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
