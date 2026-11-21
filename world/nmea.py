"""NMEA 0183 messages and UDP output for external navigation displays."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import socket

import numpy as np

from ._geo import TimeLike, normalise_time

KNOTS_PER_METRE_PER_SECOND = 1.9438444924406


def nmea_checksum(payload: str) -> int:
    """Return the NMEA 0183 XOR checksum for a sentence body."""

    checksum = 0
    for character in payload:
        checksum ^= ord(character)
    return checksum


def nmea_sentence(payload: str) -> str:
    """Wrap an NMEA sentence body with ``$``, checksum, and CR/LF."""

    body = payload.strip()
    if body.startswith("$"):
        body = body[1:]
    if "*" in body:
        raise ValueError("Provide an NMEA sentence body without a checksum.")
    if not body:
        raise ValueError("NMEA sentence body must not be empty.")
    return f"${body}*{nmea_checksum(body):02X}\r\n"


def ais_sentence(payload: str) -> str:
    """Wrap an AIS NMEA payload with ``!``, checksum, and CR/LF."""

    body = payload.strip()
    if body.startswith("!"):
        body = body[1:]
    if "*" in body:
        raise ValueError("Provide an AIS payload without a checksum.")
    if not body:
        raise ValueError("AIS payload must not be empty.")
    return f"!{body}*{nmea_checksum(body):02X}\r\n"


def _ais_unsigned(value: int, width: int) -> str:
    if not 0 <= value < 1 << width:
        raise ValueError(f"AIS value {value} cannot fit in {width} bits.")
    return f"{value:0{width}b}"


def _ais_signed(value: int, width: int) -> str:
    lower, upper = -(1 << (width - 1)), 1 << (width - 1)
    if not lower <= value < upper:
        raise ValueError(f"AIS signed value {value} cannot fit in {width} bits.")
    return _ais_unsigned(value % (1 << width), width)


def _ais_six_bit_payload(bits: str) -> tuple[str, int]:
    fill_bits = (-len(bits)) % 6
    padded = bits + "0" * fill_bits
    characters = []
    for start in range(0, len(padded), 6):
        value = int(padded[start : start + 6], 2)
        characters.append(chr(value + 48 if value < 40 else value + 56))
    return "".join(characters), fill_bits


def _ais_text(value: str, length: int) -> str:
    """Return the six-bit AIS text representation padded to ``length`` chars."""

    codes = []
    for character in value.upper()[:length]:
        code = ord(character)
        if 64 <= code <= 95:
            codes.append(code - 64)
        elif 32 <= code <= 63:
            codes.append(code)
        else:
            codes.append(0)
    codes.extend([32] * (length - len(codes)))
    return "".join(_ais_unsigned(code, 6) for code in codes)


def _ais_position(target) -> tuple[int, int]:
    latitude = float(target.latitude)
    longitude = float(target.longitude)
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("AIS target coordinates are outside WGS-84 bounds.")
    return int(round(longitude * 600_000.0)), int(round(latitude * 600_000.0))


def aivdm_position_report(target) -> str:
    """Encode an :class:`AISTarget` as a single-fragment AIS type-1 report.

    Type 1 provides OpenCPN with the dynamic target position, COG, SOG and
    heading.  Use :func:`aivdm_static_data_sentences` for its display name and
    vessel dimensions.
    """

    mmsi = target.mmsi
    if mmsi is None or not 1_000_000 <= int(mmsi) <= 999_999_999:
        raise ValueError("AIS type-1 reports need a valid 7- to 9-digit MMSI.")
    longitude, latitude = _ais_position(target)
    speed = 1023 if target.speed is None else int(round(max(0.0, float(target.speed)) * KNOTS_PER_METRE_PER_SECOND * 10.0))
    course = 3600 if target.course is None else int(round((float(target.course) % 360.0) * 10.0))
    heading = 511 if target.heading is None else int(round(float(target.heading) % 360.0))
    speed = min(speed, 1023)
    course = min(course, 3599) if course != 3600 else course
    heading = min(heading, 359)
    bits = "".join(
        (
            _ais_unsigned(1, 6),  # message type
            _ais_unsigned(0, 2),  # repeat indicator
            _ais_unsigned(int(mmsi), 30),
            _ais_unsigned(0, 4),  # underway using engine / default status
            _ais_signed(-128, 8),  # rate of turn unavailable
            _ais_unsigned(speed, 10),
            _ais_unsigned(0, 1),  # low position accuracy
            _ais_signed(longitude, 28),
            _ais_signed(latitude, 27),
            _ais_unsigned(course, 12),
            _ais_unsigned(heading, 9),
            _ais_unsigned(60, 6),  # UTC second unavailable
            _ais_unsigned(0, 2),  # manoeuvre indicator
            _ais_unsigned(0, 3),
            _ais_unsigned(0, 1),  # RAIM
            _ais_unsigned(0, 19),  # radio status
        )
    )
    payload, fill_bits = _ais_six_bit_payload(bits)
    return ais_sentence(f"AIVDM,1,1,,A,{payload},{fill_bits}")


def aivdm_static_data_sentences(target, *, sequence_id: str = "1") -> tuple[str, ...]:
    """Encode target metadata as the two fragments of an AIS type-5 report."""

    mmsi = target.mmsi
    if mmsi is None or not 1_000_000 <= int(mmsi) <= 999_999_999:
        raise ValueError("AIS type-5 reports need a valid 7- to 9-digit MMSI.")
    length = max(1.0, float(target.length_metres or 1.0))
    beam = max(1.0, float(target.beam_metres or 1.0))
    to_bow = min(511, int(round(length * 0.55)))
    to_stern = min(511, int(round(length)) - to_bow)
    to_port = min(63, int(round(beam / 2.0)))
    to_starboard = min(63, int(round(beam)) - to_port)
    ship_type = 0 if target.ship_type is None else int(target.ship_type)
    bits = "".join(
        (
            _ais_unsigned(5, 6),
            _ais_unsigned(0, 2),
            _ais_unsigned(int(mmsi), 30),
            _ais_unsigned(0, 2),  # AIS version
            _ais_unsigned(0, 30),  # IMO unavailable
            _ais_text(target.callsign or "SIM", 7),
            _ais_text(target.name or f"SIM {mmsi}", 20),
            _ais_unsigned(max(0, min(ship_type, 99)), 8),
            _ais_unsigned(to_bow, 9),
            _ais_unsigned(to_stern, 9),
            _ais_unsigned(to_port, 6),
            _ais_unsigned(to_starboard, 6),
            _ais_unsigned(1, 4),  # GPS position device
            _ais_unsigned(0, 4),
            _ais_unsigned(0, 5),
            _ais_unsigned(24, 5),
            _ais_unsigned(60, 6),
            _ais_unsigned(0, 8),  # draught unavailable
            _ais_text(target.destination or "SIMULATION", 20),
            _ais_unsigned(0, 1),
            _ais_unsigned(0, 1),
        )
    )
    payload, fill_bits = _ais_six_bit_payload(bits)
    if len(payload) != 71 or fill_bits != 2:
        raise RuntimeError("Unexpected AIS type-5 payload length.")
    return (
        ais_sentence(f"AIVDM,2,1,{sequence_id},A,{payload[:60]},0"),
        ais_sentence(f"AIVDM,2,2,{sequence_id},A,{payload[60:]},{fill_bits}"),
    )


def nmea_coordinate(value: float, *, latitude: bool) -> tuple[str, str]:
    """Convert decimal degrees to NMEA ``ddmm.mmmm``/``dddmm.mmmm`` fields."""

    limit = 90.0 if latitude else 180.0
    if not math.isfinite(value) or not -limit <= value <= limit:
        axis = "latitude" if latitude else "longitude"
        raise ValueError(f"{axis} must be finite and in [-{limit}, {limit}].")

    degrees = int(abs(value))
    minutes = (abs(value) - degrees) * 60.0
    # Protect valid values extremely close to the next full degree from a
    # formatted 60.0000-minute field.
    if round(minutes, 4) >= 60.0:
        degrees += 1
        minutes = 0.0
    degree_width = 2 if latitude else 3
    coordinate = f"{degrees:0{degree_width}d}{minutes:07.4f}"
    if latitude:
        hemisphere = "N" if value >= 0.0 else "S"
    else:
        hemisphere = "E" if value >= 0.0 else "W"
    return coordinate, hemisphere


def _nmea_utc_fields(when: TimeLike) -> tuple[str, str]:
    """Return NMEA UTC time and date fields from a UTC simulation timestamp."""

    nanoseconds = int(normalise_time(when).astype("datetime64[ns]").astype(np.int64))
    seconds, fraction = divmod(nanoseconds, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    centiseconds = fraction // 10_000_000
    return f"{timestamp:%H%M%S}.{centiseconds:02d}", f"{timestamp:%d%m%y}"


def rmc_sentence(
    when: TimeLike,
    latitude: float,
    longitude: float,
    speed_mps: float,
    course_true_degrees: float,
) -> str:
    """Create a valid-fix GPS RMC sentence from simulated navigation data."""

    if not math.isfinite(speed_mps) or speed_mps < 0.0:
        raise ValueError("speed_mps must be finite and non-negative.")
    if not math.isfinite(course_true_degrees):
        raise ValueError("course_true_degrees must be finite.")

    time_field, date_field = _nmea_utc_fields(when)
    latitude_field, latitude_hemisphere = nmea_coordinate(latitude, latitude=True)
    longitude_field, longitude_hemisphere = nmea_coordinate(longitude, latitude=False)
    speed_knots = speed_mps * KNOTS_PER_METRE_PER_SECOND
    course = course_true_degrees % 360.0
    return nmea_sentence(
        f"GPRMC,{time_field},A,{latitude_field},{latitude_hemisphere},"
        f"{longitude_field},{longitude_hemisphere},{speed_knots:.2f},{course:.1f},"
        f"{date_field},,,A"
    )


def hdt_sentence(heading_true_degrees: float) -> str:
    """Create a true-heading sentence for the simulated hull orientation."""

    if not math.isfinite(heading_true_degrees):
        raise ValueError("heading_true_degrees must be finite.")
    return nmea_sentence(f"HCHDT,{heading_true_degrees % 360.0:.1f},T")


class NMEA0183UDPOutput:
    """Send complete NMEA 0183 sentences to a UDP listener such as OpenCPN."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10110) -> None:
        if not host:
            raise ValueError("host must not be empty.")
        if not 1 <= port <= 65_535:
            raise ValueError("port must be in [1, 65535].")
        self.destination = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, *sentences: str) -> None:
        """Transmit each complete sentence in its own UDP datagram."""

        for sentence in sentences:
            self._socket.sendto(sentence.encode("ascii"), self.destination)

    def close(self) -> None:
        """Release the UDP socket."""

        self._socket.close()

    def __enter__(self) -> "NMEA0183UDPOutput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class NMEA0183TCPClientOutput:
    """Send NMEA 0183 sentences to an existing TCP listener such as OpenCPN."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10110) -> None:
        if not host:
            raise ValueError("host must not be empty.")
        if not 1 <= port <= 65_535:
            raise ValueError("port must be in [1, 65535].")
        self.destination = (host, port)
        self._socket = socket.create_connection(self.destination)

    def send(self, *sentences: str) -> None:
        """Transmit complete sentences on the established TCP connection."""

        if sentences:
            self._socket.sendall("".join(sentences).encode("ascii"))

    def close(self) -> None:
        """Release the TCP connection."""

        self._socket.close()

    def __enter__(self) -> "NMEA0183TCPClientOutput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class NMEA0183TCPServerOutput:
    """Accept one OpenCPN TCP-input connection and stream NMEA sentences."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10110) -> None:
        if not host:
            raise ValueError("host must not be empty.")
        if not 1 <= port <= 65_535:
            raise ValueError("port must be in [1, 65535].")
        self.destination = (host, port)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(self.destination)
        self._listener.listen(1)
        self._connection: socket.socket | None = None

    def wait_for_client(self) -> None:
        """Wait for OpenCPN to connect to the configured local TCP endpoint."""

        self._connection, _ = self._listener.accept()

    def send(self, *sentences: str) -> None:
        """Transmit complete sentences to the connected OpenCPN client."""

        if self._connection is None:
            raise RuntimeError("No TCP client is connected. Call wait_for_client() first.")
        if sentences:
            self._connection.sendall("".join(sentences).encode("ascii"))

    def close(self) -> None:
        """Release the connected client and listening sockets."""

        if self._connection is not None:
            self._connection.close()
        self._listener.close()

    def __enter__(self) -> "NMEA0183TCPServerOutput":
        self.wait_for_client()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
