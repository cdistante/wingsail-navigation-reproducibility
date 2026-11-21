"""
Download Copernicus Marine wave forecasts
for the Mediterranean Sea.

Author:
Cosimo Distante

Output:

data/waves/

wave_med_YYYYMMDD.nc
"""

import argparse
import os
import sys

from pathlib import Path

from datetime import datetime, UTC, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import WAVES


# Mediterranean

LAT_MIN = 30.0
LAT_MAX = 46.0

LON_MIN = -6.0
LON_MAX = 36.0

import copernicusmarine

# Copernicus Marine Mediterranean wave analysis/forecast dataset.
# The previous ID ending in "-m" is not found in the current catalogue.
DATASET_ID = "cmems_mod_med_wav_anfc_4.2km_PT1H-i"
DEFAULT_FORECAST_HOURS = 72

VARIABLES = [
    "VHM0",  # Spectral significant wave height
    "VMDR",  # Mean wave direction
    "VTPK",  # Wave period at spectral peak
]


def login_to_copernicus() -> None:
    username = os.getenv("CMEMS_USER")
    password = os.getenv("CMEMS_PASSWORD")

    if username and password:
        copernicusmarine.login(
            username=username,
            password=password,
        )
        print("Logged in to Copernicus Marine using environment variables.")
        return

    print(
        "CMEMS_USER and CMEMS_PASSWORD are not set. "
        "Continuing with stored Copernicus Marine credentials, if available."
    )


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalise it to UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def download(
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    *,
    start_datetime: datetime | None = None,
) -> Path:
    if forecast_hours <= 0:
        raise ValueError("forecast_hours must be positive.")
    WAVES.mkdir(exist_ok=True, parents=True)

    retrieval_datetime = datetime.now(UTC)
    start_datetime = start_datetime or retrieval_datetime
    if start_datetime.tzinfo is None:
        start_datetime = start_datetime.replace(tzinfo=UTC)
    start_datetime = start_datetime.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    )
    retrieval_time = start_datetime.strftime("%Y%m%d_%H")

    outfile = WAVES / f"wave_med_{retrieval_time}.nc"

    # Keep a multi-day horizon so a full route simulation does not run out of
    # sea-state data after the first day.
    end_datetime = start_datetime + timedelta(hours=forecast_hours)

    print()
    print("------------------------------------")
    print("Downloading Copernicus Wave Forecast")
    print("------------------------------------")
    print(f"Dataset: {DATASET_ID}")
    print(f"Output:  {outfile}")
    print(f"Horizon: {forecast_hours} hours")
    print(f"Time:    {start_datetime.isoformat()} to {end_datetime.isoformat()}")

    try:
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=VARIABLES,
            minimum_latitude=LAT_MIN,
            maximum_latitude=LAT_MAX,
            minimum_longitude=LON_MIN,
            maximum_longitude=LON_MAX,
            start_datetime=start_datetime.isoformat(),
            end_datetime=end_datetime.isoformat(),
            output_filename=outfile.name,
            output_directory=str(WAVES),
            force_download=True,
        )
    except copernicusmarine.catalogue_parser.models.DatasetNotFound as exc:
        raise RuntimeError(
            f"Copernicus Marine dataset was not found: {DATASET_ID}. "
            "The dataset ID may have changed in the Copernicus catalogue. "
            "Check the current dataset ID in the Copernicus Marine Data Store "
            "and update DATASET_ID in this script."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Copernicus Marine wave data. "
            "If this is an authentication error, set the CMEMS_USER and "
            "CMEMS_PASSWORD environment variables in your PyCharm run configuration, "
            "or log in once using the Copernicus Marine CLI."
        ) from exc

    print()
    print("Finished.")
    print(outfile)

    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a time-varying Mediterranean wave forecast."
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=DEFAULT_FORECAST_HOURS,
        help="Forecast horizon to download (default: 72).",
    )
    parser.add_argument(
        "--start-time",
        help=(
            "Optional ISO-8601 UTC start, for example 2026-09-30T00:00:00Z. "
            "Defaults to the current hour."
        ),
    )
    arguments = parser.parse_args()
    login_to_copernicus()
    download(
        arguments.forecast_hours,
        start_datetime=parse_utc(arguments.start_time) if arguments.start_time else None,
    )
