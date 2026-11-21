"""Live AIS receiver for an OpenCPN NMEA multiplexer TCP output."""

from __future__ import annotations

import socket
from dataclasses import replace
from time import monotonic
from typing import Any

from .ais import AISTarget
from .nmea import KNOTS_PER_METRE_PER_SECOND


class LiveAIS:
    """Decode live ``!AIVDM`` sentences from an OpenCPN TCP connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 3000) -> None:
        self.host = host
        self.port = int(port)
        self._socket: socket.socket | None = None
        self._buffer = b""
        self._targets: dict[int, AISTarget] = {}
        self._received_at: dict[int, float] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def time_mode(self) -> str:
        return "static"

    def load(self) -> "LiveAIS":
        """Connect to the multiplexer and make the socket nonblocking."""

        self.close()
        self._socket = socket.create_connection((self.host, self.port), timeout=10.0)
        self._socket.setblocking(False)
        self._loaded = True
        return self

    def targets(self, when=None) -> list[AISTarget]:
        """Poll available NMEA data and return the latest target reports."""

        if not self._loaded or self._socket is None:
            raise RuntimeError("Live AIS is not loaded. Call load() first.")
        self._poll()
        now = monotonic()
        return [
            replace(target, report_age_seconds=now - self._received_at[mmsi])
            for mmsi, target in self._targets.items()
        ]

    def sample(self, when=None) -> list[AISTarget]:
        return self.targets(when)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._buffer = b""
        self._targets.clear()
        self._received_at.clear()
        self._loaded = False

    def _poll(self) -> None:
        assert self._socket is not None
        while True:
            try:
                chunk = self._socket.recv(65536)
            except (BlockingIOError, socket.timeout):
                break
            if not chunk:
                self.close()
                break
            self._buffer += chunk

        while b"\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split(b"\n", 1)
            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line.startswith("!AI"):
                continue
            try:
                target = self._decode(line)
            except Exception:
                continue
            if target.mmsi is not None:
                self._targets[target.mmsi] = target
                self._received_at[target.mmsi] = monotonic()

    @staticmethod
    def _decode(line: str) -> AISTarget:
        from pyais import decode

        values: dict[str, Any] = decode(line).asdict()

        def value(*names: str):
            for name in names:
                if values.get(name) is not None:
                    return values[name]
            return None

        def number(*names: str) -> float | None:
            raw = value(*names)
            return None if raw is None else float(raw)

        speed_knots = number("speed", "sog")
        bow, stern = number("dim_a", "to_bow"), number("dim_b", "to_stern")
        port, starboard = number("dim_c", "to_port"), number("dim_d", "to_starboard", "to_stbd")

        raw_mmsi = value("mmsi")
        return AISTarget(
            mmsi=None if raw_mmsi is None else int(raw_mmsi),
            latitude=float(value("lat", "latitude")),
            longitude=float(value("lon", "longitude")),
            speed=(
                None
                if speed_knots is None
                else speed_knots / KNOTS_PER_METRE_PER_SECOND
            ),
            course=number("course", "cog"),
            heading=number("heading", "true_heading"),
            name=value("shipname", "name"),
            callsign=value("callsign", "call_sign"),
            ship_type=None if value("ship_type", "shiptype") is None else int(value("ship_type", "shiptype")),
            length_metres=None if bow is None or stern is None else bow + stern,
            beam_metres=None if port is None or starboard is None else port + starboard,
            destination=value("destination"),
        )


class LiveAISUDP(LiveAIS):
    """Decode live ``!AIVDM`` datagrams from an OpenCPN UDP output."""

    def load(self) -> "LiveAISUDP":
        """Bind the local UDP endpoint and make the socket nonblocking."""

        self.close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)
        self._loaded = True
        return self

    def _poll(self) -> None:
        assert self._socket is not None
        while True:
            try:
                chunk, _ = self._socket.recvfrom(65536)
            except (BlockingIOError, socket.timeout):
                break
            if not chunk:
                break
            self._buffer += chunk

        super()._poll()
