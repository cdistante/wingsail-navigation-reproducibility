"""Start the hybrid wave with its validation-selection and runtime hashes pinned."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

import launch_hybrid as paired


VALIDATION_SELECTION = paired.STUDY.with_name("learned_weather_ais_hybrid") / "validation_selection.json"
model_source_hashes = paired.model_sources


def checkpoint_and_selection_hashes(model, digest):
    sources = model_source_hashes(model, digest)
    selection = json.loads(VALIDATION_SELECTION.read_text())
    assert selection["minimum_forecast_gain_s"] == 180.0, "The validation-selected veto threshold changed."
    critic_name = str((model/"joint_critic.npz").relative_to(paired.ROOT))
    assert selection["learner_checkpoint_sha256"] == sources[critic_name], "The checkpoint differs from validation."
    for path in (VALIDATION_SELECTION, Path(__file__), paired.ROOT / "world/learned_mark_policy.py"):
        sources[str(path.relative_to(paired.ROOT))] = digest(path)
    return sources


if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "hybrid"]
    with patch.object(paired, "model_sources", checkpoint_and_selection_hashes):
        paired.main()
