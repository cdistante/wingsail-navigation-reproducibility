"""Create viewer replays with correct method labels and unchanged frame data."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil


STUDY = Path(__file__).resolve().parents[1]
PHASE_LINE = "Refinement on return · outbound uses original 120 s scheduler"
PERIOD_LINE = "Forecast replanning every 60 s"
REGATTA_LINE = "Regatta guidance · forecast time to mark + refined tacks"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def body_digest(path):
    with gzip.open(path, "rb") as stream:
        next(stream)
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label(directory):
    execution = json.loads((directory / "execution.json").read_text())
    on = "--regatta-time-to-mark" in execution["command"]
    mode = "on" if on else "off"
    assert execution["regatta_mode"] == mode
    raw = directory / "navigation.jsonl.gz"
    playback = directory / "navigation_playback.jsonl.gz"
    pending = playback.with_suffix(".tmp")
    with gzip.open(raw, "rb") as source:
        header = json.loads(next(source))
        lines = header["method_lines"]
        assert lines.count(PERIOD_LINE) == lines.count(PHASE_LINE) == 1, directory
        assert lines.count(REGATTA_LINE) == int(on), directory
        if on:
            lines[lines.index(REGATTA_LINE)] = "Regatta ON · 10 h holding time-to-mark forecast"
            lines[lines.index(PERIOD_LINE)] = "Tack scheduler · 60 s on outbound and return"
            lines[lines.index(PHASE_LINE)] = "Tack refinement · outbound and return"
        else:
            lines.insert(lines.index("Forecast tack planning"), "Regatta OFF · 3 h forecast hold")
            lines[lines.index(PERIOD_LINE)] = "Tack scheduler · 120 s outbound, 60 s return"
        with gzip.open(pending, "wb", compresslevel=3) as target:
            target.write((json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            shutil.copyfileobj(source, target)
    data_hash = body_digest(raw)
    assert body_digest(pending) == data_hash, directory
    pending.replace(playback)
    return {
        "regatta_mode": mode,
        "native_replay_sha256": digest(raw),
        "playback_replay_sha256": digest(playback),
        "unchanged_frame_data_sha256": data_hash,
    }


def main():
    folders = sorted((STUDY / "runs").iterdir())
    assert len(folders) == 16 and all((path / "execution.json").is_file() for path in folders)
    rows = {folder.name: label(folder) for folder in folders}
    output = STUDY / "playback_replay_headers.json"
    output.write_text(json.dumps({"runs": rows, "only_method_header_changed": True}, indent=2) + "\n")
    print(f"Verified header-only replay variants for {len(rows)} runs.")


if __name__ == "__main__":
    main()
