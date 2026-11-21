"""Fingerprint every loaded regional marine field and coordinate used here."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

from run_archive import OTRANTO_BOUNDS
from world import Currents, Waves, Weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Compare with the released receipt without writing it.")
    args = parser.parse_args()
    fields = {}
    for name, factory in (("wind", Weather), ("current", Currents), ("wave", Waves)):
        provider = factory(**OTRANTO_BOUNDS).load()
        dataset = provider._require_dataset()
        variables = {}
        for key in sorted(dataset.variables):
            value = dataset[key]
            array = np.asarray(value.values)
            if array.dtype.kind == "O":
                array = array.astype(str)
            contiguous = np.ascontiguousarray(array)
            variables[key] = {"dims": list(value.dims), "shape": list(array.shape), "dtype": array.dtype.str,
                              "sha256": hashlib.sha256(memoryview(contiguous.view(np.uint8)).cast("B")).hexdigest()}
        fields[name] = {"variables": variables,
                        "sha256": hashlib.sha256(json.dumps(variables, sort_keys=True).encode()).hexdigest()}
        provider.close()
    result = {"regional_bounds": OTRANTO_BOUNDS, "fields": fields,
              "meaning": "All loaded regional archive variable arrays and coordinates, not only capture samples.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = STUDY / "archive_field_fingerprints.json"
    if args.check:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if expected["regional_bounds"] != {name: list(bounds) for name, bounds in OTRANTO_BOUNDS.items()}:
            raise SystemExit("Loaded regional bounds differ from the archived receipt.")
        mismatches = [name for name in fields if fields[name] != expected["fields"].get(name)]
        if mismatches:
            raise SystemExit(f"Loaded archive fields differ: {', '.join(mismatches)}.")
        count = sum(len(field["variables"]) for field in fields.values())
        print(f"All {count} loaded regional variables and coordinates match the archived receipt.")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
