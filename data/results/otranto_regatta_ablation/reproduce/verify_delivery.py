"""Bind the completed simulations, independent measurements, media and manuscript."""

import hashlib
import json
from pathlib import Path
import re
import zipfile

from label_playback_replays import body_digest


STUDY = Path(__file__).resolve().parents[1]
PREVIOUS = STUDY.with_name("otranto_underway_ais_retest")
PAPER = STUDY.parents[2] / "reports/journal_manuscript"
ORDER = [("outbound", mark) for mark in ("002", "003", "004", "005")] + [
    ("return", mark) for mark in ("004", "003", "002", "001")
]
RECEIPTS = (
    "comparison.json", "independent_comparison.json", "independent_comparison.csv",
    "independent_arm_outcomes.csv", "independent_acceptance_check.json",
    "independent_final_summary.json", "independent_waypoint_deltas.csv",
    "independent_inputs_verification.json", "independent_traffic_sweep.json",
    "launch_preflight.json", "launch_gate_verification.json", "launch_physics_verification.json",
    "media_verification.json", "media_layout_verification.json", "playback.md", "playback_replay_headers.json",
    "paper_layout_verification.json", "paper_package_verification.json", "paper_native_compile.json",
    "audit_outputs_sha256.json", "paired_regatta_leg_deltas.csv",
)


def read(name):
    return json.loads((STUDY / name).read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_paper():
    layout = read("paper_layout_verification.json")
    package = read("paper_package_verification.json")
    assert layout["all_pages_rendered_and_visually_reviewed"]
    assert read("paper_native_compile.json")["all_compile_checks_passed"]
    assert digest(PAPER / "manuscript.pdf") == layout["pdf_sha256"]
    sources = PAPER / "journal_manuscript_sources.zip"
    assert digest(sources) == layout["source_zip_sha256"] == package["source_zip_sha256"]
    with zipfile.ZipFile(sources) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == package["source_zip_files"]
        assert all(archive.read(name) == (PAPER / name).read_bytes() for name in archive.namelist())
    for name, expected in read("previous_paper/sha256.json").items():
        assert digest(STUDY / "previous_paper" / name) == expected


def main():
    main_results = read("comparison.json")
    paired = read("independent_comparison.json")
    acceptance = read("independent_acceptance_check.json")
    assert main_results["complete_matrix"] and paired["complete_matrix"]
    assert main_results["radius_m"] == 50 and main_results["dt_s"] == 5
    assert read("independent_inputs_verification.json")["all_closed_verified"]
    assert read("launch_physics_verification.json")["all_pairs_pass_gate_and_runtime_option_checks"]
    runs = main_results["runs"]
    sweeps = {row["case"]: row for row in read("independent_traffic_sweep.json")}
    media = read("media_verification.json")
    playback_replays = read("playback_replay_headers.json")
    assert playback_replays["only_method_header_changed"]
    assert media["all_sixteen_mp4s_fully_decoded"]
    assert read("media_layout_verification.json")["all_final_previews_passed"]
    assert len(runs) == 16 and set(runs) == set(sweeps) == set(media["runs"]) == set(playback_replays["runs"])
    assert {row["case"] for row in acceptance["cases"]} == set(runs)
    completed = captures = 0
    for case, run in runs.items():
        marks = run["captures"]
        order = [(entry["phase"], entry["waypoint"]) for entry in marks]
        assert order == ORDER[:len(order)] and all(mark["distance_metres"] <= 50 for mark in marks)
        if run["status"] == "completed":
            assert order == ORDER
            completed += 1
        captures += len(order)
        folder = STUDY / "runs" / case
        info = media["runs"][case]
        labels = playback_replays["runs"][case]
        raw = folder / "navigation.jsonl.gz"
        viewer = folder / "navigation_playback.jsonl.gz"
        assert digest(raw) == sweeps[case]["replay_sha256"] == info["replay_sha256"] == labels["native_replay_sha256"]
        assert digest(viewer) == info["playback_replay_sha256"] == labels["playback_replay_sha256"]
        assert body_digest(raw) == body_digest(viewer) == labels["unchanged_frame_data_sha256"]
        assert digest(folder / "navigation.mp4") == info["video_sha256"]
        assert (folder / "track.gpx").is_file() and (folder / "departure_gate.jsonl").is_file()
    measurements = read("independent_final_summary.json")
    assert measurements["verified_captures"] == captures and measurements["completed_loops"] == completed
    assert measurements["all16_gate_and_native_acceptance_passed"] == acceptance["all_passed"]
    rows = paired["paired_comparisons"]
    assert len(rows) == 8 and {row[key] for row in rows for key in ("on_arm", "off_arm")} == set(runs)
    for row in rows:
        assert row["identical_production_sources_and_behavior_arguments_except_flag"]
        assert row["identical_clock_forcing_and_actual_launch"]
    index = (STUDY / "playback.md").read_text()
    links = re.findall(r"\]\(([^:()]+)\)", index)
    assert len(links) >= 32 and all((STUDY / name).is_file() for name in links)
    assert digest(STUDY / "playback.md") == media["playback_index_sha256"]
    verify_paper()
    manifest = read("audit_outputs_sha256.json")
    assert manifest and all(digest(STUDY / name) == expected for name, expected in manifest.items())
    old_manifest = json.loads((PREVIOUS / "audit_outputs_sha256.json").read_text())
    assert all(digest(PREVIOUS / name) == expected for name, expected in old_manifest.items())
    previous_media = json.loads((PREVIOUS / "media_verification.json").read_text())["runs"]
    assert all(digest(PREVIOUS / "runs" / case / "navigation.mp4") == info["video_sha256"]
               for case, info in previous_media.items())
    result = {
        "verified_fresh_runs": len(runs),
        "verified_pairs": len(rows),
        "completed_loops": completed,
        "captures_within_50m_in_order": captures,
        "all_independent_acceptance_checks_passed": acceptance["all_passed"],
        "verified_replay_and_mp4_sets": len(media["runs"]),
        "verified_playback_links": len(links),
        "previous_study_manifest_entries_preserved": len(old_manifest),
        "previous_videos_preserved": len(previous_media),
        "frozen_manifest_entries": len(manifest),
        "receipts_sha256": {name: digest(STUDY / name) for name in RECEIPTS},
    }
    (STUDY / "deliverable_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: value for name, value in result.items() if name != "receipts_sha256"}, indent=2))


if __name__ == "__main__":
    main()
