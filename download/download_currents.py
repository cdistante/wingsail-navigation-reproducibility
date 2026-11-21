"""
Download HYCOM ocean currents
for the Mediterranean Sea.

Author:
Cosimo Distante

Output:

data/currents/

hycom_currents_YYYYMMDD.nc
"""

import argparse
from pathlib import Path
from datetime import datetime, UTC, timedelta
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from config import CURRENTS


# ------------------------------------------------------------
# Mediterranean bounding box
# ------------------------------------------------------------

LAT_MIN = 30.0
LAT_MAX = 46.0

LON_MIN = -6.0
LON_MAX = 36.0


# ------------------------------------------------------------
# HYCOM THREDDS NetCDF Subset Service
# ------------------------------------------------------------

HYCOM_NCSS_URL = (
    "https://tds.hycom.org/thredds/ncss/grid/"
    "FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
)

REQUEST_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10

DEFAULT_FORECAST_HOURS = 72


def _response_preview(response: requests.Response, max_chars: int = 500) -> str:
    """
    Return a short text preview of an HTTP response body.

    This is useful when THREDDS returns an HTML/XML error page instead
    of a NetCDF file.
    """
    try:
        text = response.text.strip()
    except Exception:
        return "<response body could not be decoded>"

    if not text:
        return "<empty response body>"

    return text[:max_chars]


def _validate_response_is_netcdf(response: requests.Response) -> None:
    """
    Validate that the HYCOM server returned something compatible with NetCDF.

    THREDDS sometimes returns XML/HTML error documents with HTTP 200,
    so checking only the status code is not enough.
    """
    content_type = response.headers.get("Content-Type", "").lower()

    invalid_content_types = (
        "text/html",
        "text/plain",
        "application/xml",
        "text/xml",
    )

    if any(value in content_type for value in invalid_content_types):
        preview = _response_preview(response)
        raise RuntimeError(
            "HYCOM returned an error document instead of a NetCDF file.\n"
            f"Content-Type: {content_type or '<missing>'}\n"
            f"Server response preview:\n{preview}\n\n"
            "This usually means the HYCOM dataset URL, variable names, "
            "or subset parameters are no longer accepted by the server."
        )


def _build_params(start_time: datetime, end_time: datetime) -> dict:
    """
    Build HYCOM NCSS request parameters.

    The ``best`` FMRC aggregation accepts a time range and returns every
    available surface-current snapshot in that interval.  The world layer can
    then interpolate the resulting time series at each simulator step.
    """
    return {
        "var": ["water_u", "water_v"],
        "north": LAT_MAX,
        "south": LAT_MIN,
        "west": LON_MIN,
        "east": LON_MAX,
        "disableProjSubset": "on",
        "horizStride": 1,
        "time_start": start_time.isoformat(),
        "time_end": end_time.isoformat(),
        "timeStride": 1,
        "vertCoord": 0,
        "accept": "netcdf",
    }


def download(forecast_hours: int = DEFAULT_FORECAST_HOURS) -> Path:
    if forecast_hours <= 0:
        raise ValueError("forecast_hours must be positive.")
    CURRENTS.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=forecast_hours)
    retrieval_time = start_time.strftime("%Y%m%d_%H")

    outfile = CURRENTS / f"hycom_currents_{retrieval_time}.nc"
    temporary_outfile = outfile.with_suffix(outfile.suffix + ".part")

    print("---------------------------------------")
    print("Downloading HYCOM Mediterranean currents...")
    print("---------------------------------------")
    print(f"URL:    {HYCOM_NCSS_URL}")
    print(f"Output: {outfile}")
    print(f"Time:   {start_time.isoformat()} to {end_time.isoformat()}")

    last_error: Exception | None = None

    params = _build_params(start_time, end_time)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print()
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}...")

        try:
            if temporary_outfile.exists():
                temporary_outfile.unlink()

            with requests.get(
                HYCOM_NCSS_URL,
                params=params,
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    preview = _response_preview(response)
                    raise RuntimeError(
                        "HYCOM server returned an HTTP error.\n"
                        f"Status code: {response.status_code}\n"
                        f"Final URL: {response.url}\n"
                        f"Server response preview:\n{preview}"
                    ) from exc

                _validate_response_is_netcdf(response)

                bytes_written = 0

                with temporary_outfile.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
                            bytes_written += len(chunk)

                if bytes_written == 0:
                    raise RuntimeError(
                        f"HYCOM download failed: downloaded file is empty: {temporary_outfile}"
                    )

                temporary_outfile.replace(outfile)

                print()
                print("---------------------------------------")
                print("Saved")
                print(outfile)
                print(f"Size: {outfile.stat().st_size:,} bytes")
                print("---------------------------------------")

                return outfile

        except requests.RequestException as exc:
            last_error = exc
            print(f"Network/request error: {exc}")

        except RuntimeError as exc:
            last_error = exc
            print(f"Download validation error: {exc}")

        if attempt < MAX_ATTEMPTS:
            print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)

    if temporary_outfile.exists():
        temporary_outfile.unlink()

    raise RuntimeError(
        "Failed to download HYCOM current data after multiple attempts.\n\n"
        "Most likely causes:\n"
        "1. The HYCOM THREDDS server is temporarily unavailable.\n"
        "2. The HYCOM dataset URL has changed.\n"
        "3. The requested variables or subset parameters are no longer valid.\n"
        "4. Your network connection blocked or timed out the request.\n\n"
        "The HYCOM server may not yet expose the requested forecast window, "
        "or its NCSS query syntax/dataset URL may have changed.\n\n"
        f"Last error:\n{last_error}"
    ) from last_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a time-varying HYCOM Mediterranean surface-current forecast."
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=DEFAULT_FORECAST_HOURS,
        help="Forecast horizon to download (default: 72).",
    )
    arguments = parser.parse_args()
    download(arguments.forecast_hours)
