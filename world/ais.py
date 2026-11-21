"""AIS replay layer for dynamic vessel traffic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from config import AIS_PROCESSED

from ._geo import TimeLike, normalise_time


@dataclass(frozen=True)
class AISTarget:
    """One traffic target in the convention used by the simulator."""

    mmsi: int | None
    latitude: float
    longitude: float
    speed: float | None
    course: float | None
    heading: float | None
    name: str | None = None
    callsign: str | None = None
    ship_type: int | None = None
    length_metres: float | None = None
    beam_metres: float | None = None
    destination: str | None = None
    report_age_seconds: float | None = None


class AIS:
    """Replay processed AIS CSV tracks against the simulation clock.

    A CSV with ISO-8601 timestamps is replayed against UTC timestamps.  A CSV
    with numeric timestamps is replayed against elapsed simulation seconds.
    The current ``download_ais.py`` emits numeric NMEA seconds, so it is
    treated as an elapsed-time stream until a full receiver timestamp is
    available.
    """

    _COLUMN_ALIASES = {
        "latitude": "latitude",
        "lat": "latitude",
        "longitude": "longitude",
        "lon": "longitude",
        "mmsi": "mmsi",
        "speed": "speed",
        "sog": "speed",
        "course": "course",
        "cog": "course",
        "heading": "heading",
        "ship_name": "name",
        "receiver_timestamp": "timestamp",
        "timestamp": "timestamp",
        "time": "timestamp",
    }

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        mmsi: int | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.mmsi = None if mmsi is None else int(mmsi)
        self._tracks: dict[object, pd.DataFrame] = {}
        self._time_mode: str = "static"

    @property
    def loaded(self) -> bool:
        return bool(self._tracks)

    @property
    def time_mode(self) -> str:
        """Return ``static``, ``elapsed_seconds``, or ``utc_datetime``."""

        return self._time_mode

    def load(self, path: Path | str | None = None) -> "AIS":
        """Load a processed AIS CSV file into per-vessel tracks."""

        self.close()
        if path is not None:
            self.path = Path(path)
        source = self.path or self._find_default_file()
        if not source.exists():
            raise RuntimeError(f"AIS replay file not found: {source}.")

        frame = pd.read_csv(source)
        frame = self._normalise_columns(frame, source)
        frame = frame.dropna(subset=["latitude", "longitude"])
        if frame.empty:
            raise ValueError(f"AIS replay contains no valid positions: {source}.")

        self.path = source
        self._time_mode = self._prepare_time(frame)
        key = frame["mmsi"] if "mmsi" in frame else pd.Series(np.arange(len(frame)), index=frame.index)
        self._tracks = {
            mmsi: group.sort_values("_time").reset_index(drop=True)
            for mmsi, group in frame.groupby(key, dropna=False, sort=False)
        }
        if self.mmsi is not None:
            self._tracks = {
                track_mmsi: track
                for track_mmsi, track in self._tracks.items()
                if not pd.isna(track_mmsi) and int(track_mmsi) == self.mmsi
            }
            if not self._tracks:
                raise ValueError(f"MMSI {self.mmsi} is not present in AIS replay: {source}.")
        return self

    def targets(self, when: TimeLike | float | int | None = None) -> list[AISTarget]:
        """Return the most recent report for every vessel at ``when``.

        When no timestamp is available, all final reports are returned.  This
        is useful for static synthetic traffic as well as malformed legacy
        AIS files.
        """

        if not self._tracks:
            raise RuntimeError("AIS is not loaded. Call load() first.")

        query_time = self._normalise_query_time(when)
        result: list[AISTarget] = []
        for track in self._tracks.values():
            row = self._select_report(track, query_time)
            if row is not None:
                age = None
                if query_time is not None:
                    if self._time_mode == "elapsed_seconds":
                        age = float(query_time - row["_time"])
                    else:
                        age = float((pd.Timestamp(query_time) - row["_time"]).total_seconds())
                result.append(self._to_target(row, age))
        return result

    def sample(self, when: TimeLike | float | int | None = None) -> list[AISTarget]:
        """Alias for :meth:`targets` to match the world-layer interface."""

        return self.targets(when)

    def info(self) -> dict[str, object] | None:
        """Return replay metadata."""

        if not self._tracks or self.path is None:
            return None
        reports = sum(len(track) for track in self._tracks.values())
        return {
            "path": str(self.path),
            "vessels": len(self._tracks),
            "reports": reports,
            "time_mode": self._time_mode,
        }

    def close(self) -> None:
        """Clear all loaded tracks."""

        self._tracks = {}
        self._time_mode = "static"

    def _find_default_file(self) -> Path:
        files = sorted(AIS_PROCESSED.glob("*.csv"))
        if not files:
            raise RuntimeError(
                f"No processed AIS CSV found in {AIS_PROCESSED}. Run download/download_ais.py first."
            )
        return files[0]

    def _normalise_columns(self, frame: pd.DataFrame, source: Path) -> pd.DataFrame:
        renames = {
            name: self._COLUMN_ALIASES[name.lower()]
            for name in frame.columns
            if name.lower() in self._COLUMN_ALIASES
        }
        frame = frame.rename(columns=renames).copy()
        required = {"latitude", "longitude"}
        missing = required.difference(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{source.name} is missing required column(s): {names}.")

        for name in ("latitude", "longitude", "speed", "course", "heading", "length_metres", "beam_metres"):
            if name in frame:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
        return frame

    def _prepare_time(self, frame: pd.DataFrame) -> str:
        if "timestamp" not in frame:
            frame["_time"] = 0.0
            return "static"

        if pd.api.types.is_numeric_dtype(frame["timestamp"]):
            frame["_time"] = pd.to_numeric(frame["timestamp"], errors="coerce")
            if frame["_time"].isna().all():
                frame["_time"] = 0.0
                return "static"
            frame["_time"] = frame["_time"].ffill().fillna(0.0)
            return "elapsed_seconds"

        parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if parsed.isna().all():
            frame["_time"] = 0.0
            return "static"
        frame["_time"] = parsed.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        return "utc_datetime"

    def _normalise_query_time(self, when: TimeLike | float | int | None):
        if self._time_mode == "static" or when is None:
            return None
        if self._time_mode == "elapsed_seconds":
            if not isinstance(when, (float, int, np.floating, np.integer)):
                raise TypeError("Numeric AIS timestamps require elapsed simulation seconds.")
            return float(when)
        return normalise_time(when)

    def _select_report(self, track: pd.DataFrame, query_time):
        if query_time is None:
            return track.iloc[-1]

        times = track["_time"].to_numpy()
        index = int(np.searchsorted(times, query_time, side="right") - 1)
        if index < 0:
            return None
        return track.iloc[index]

    @staticmethod
    def _to_target(row: pd.Series, report_age_seconds: float | None = None) -> AISTarget:
        def optional_float(name: str) -> float | None:
            value = row.get(name)
            return None if value is None or pd.isna(value) else float(value)

        mmsi = row.get("mmsi")
        name = row.get("name")
        return AISTarget(
            mmsi=None if mmsi is None or pd.isna(mmsi) else int(mmsi),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            speed=optional_float("speed"),
            course=optional_float("course"),
            heading=optional_float("heading"),
            name=None if pd.isna(name) else str(name).strip(),
            length_metres=optional_float("length_metres"),
            beam_metres=optional_float("beam_metres"),
            report_age_seconds=report_age_seconds,
        )
