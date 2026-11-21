"""Small geographic and time utilities shared by world layers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

import numpy as np


TimeLike = Union[datetime, np.datetime64, str]

EARTH_RADIUS_M = 6_371_008.8
METRES_PER_DEGREE_LATITUDE = 111_320.0


def normalise_longitude(longitude):
    """Map longitude values to the ``[-180, 180)`` convention."""

    return ((longitude + 180.0) % 360.0) - 180.0


def normalise_time(when: TimeLike) -> np.datetime64:
    """Convert a supported time input to a UTC ``datetime64[ns]`` value."""

    if isinstance(when, datetime):
        if when.tzinfo is not None:
            when = when.astimezone(timezone.utc).replace(tzinfo=None)
        value = np.datetime64(when, "ns")
    elif isinstance(when, np.datetime64):
        value = when.astype("datetime64[ns]")
    elif isinstance(when, str):
        text = when.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            value = normalise_time(datetime.fromisoformat(text))
        except ValueError:
            value = np.datetime64(when, "ns")
    else:
        raise TypeError("time must be a datetime, numpy.datetime64, or ISO-8601 string.")

    if np.isnat(value):
        raise ValueError("time must be valid.")

    return value


def haversine_metres(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return geodesic distance between two WGS-84 positions in metres."""

    lat_a, lon_a, lat_b, lon_b = np.radians(
        [latitude_a, longitude_a, latitude_b, longitude_b]
    )
    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a
    haversine = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(
        d_lon / 2.0
    ) ** 2
    return float(2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine)))


def local_east_north_metres(
    latitude,
    longitude,
    origin_latitude: float,
    origin_longitude: float,
):
    """Project nearby WGS-84 points to an east/north local tangent plane."""

    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    latitude_delta = latitude - origin_latitude
    longitude_delta = normalise_longitude(longitude - origin_longitude)
    east = longitude_delta * METRES_PER_DEGREE_LATITUDE * np.cos(
        np.radians(origin_latitude)
    )
    north = latitude_delta * METRES_PER_DEGREE_LATITUDE
    return east, north


def offset_latitude_longitude(
    origin_latitude: float,
    origin_longitude: float,
    east_metres,
    north_metres,
):
    """Convert nearby local east/north offsets back to WGS-84 coordinates."""

    latitude = origin_latitude + np.asarray(north_metres) / METRES_PER_DEGREE_LATITUDE
    longitude = normalise_longitude(
        origin_longitude
        + np.asarray(east_metres)
        / (METRES_PER_DEGREE_LATITUDE * np.cos(np.radians(origin_latitude)))
    )
    return latitude, longitude
