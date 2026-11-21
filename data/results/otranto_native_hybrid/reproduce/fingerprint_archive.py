"""Fingerprint every loaded regional marine field and coordinate used here."""

from __future__ import annotations

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
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
