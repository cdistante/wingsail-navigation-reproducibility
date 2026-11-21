"""
Download GFS weather forecasts cropped to the Mediterranean Sea.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import urllib3

from herbie import Herbie

# Disabilita gli avvisi SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import WEATHER

DEFAULT_MAX_LEAD_HOURS = 72
DEFAULT_INTERVAL_HOURS = 3

# Limiti geografici del Mar Mediterraneo (Bounding Box)
LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -6.0, 36.0


import xarray as xr
from datetime import timedelta


def download_forecast(hour):
    print(f"\n--- Processing forecast +{hour:02d} h ---")

    now_utc = datetime.now(timezone.utc)
    run_hour = (now_utc.hour // 6) * 6
    gfs_run_date = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0, tzinfo=None)

    for attempt in range(3):
        try:
            H = Herbie(
                date=gfs_run_date,
                model="gfs",
                product="pgrb2.0p25",
                fxx=hour,
                priority=["aws", "google", "nomads"]
            )

            # Usa 'search' e aggiungi 'backend_kwargs' per evitare la separazione in hypercube multipli
            ds_list = H.xarray(
                search=":UGRD:10 m|:VGRD:10 m",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10}}
            )

            if ds_list is None:
                raise FileNotFoundError("Dati non trovati nella run.")

            # Se restituisce una lista, prendiamo il primo elemento valido
            ds = ds_list[0] if isinstance(ds_list, list) else ds_list

            # Ritaglio geografico (Slicing) sul Mediterraneo
            lon_min_gfs = LON_MIN if LON_MIN >= 0 else 360 + LON_MIN
            lon_max_gfs = LON_MAX if LON_MAX >= 0 else 360 + LON_MAX

            if lon_min_gfs > lon_max_gfs:
                ds_cropped = ds.sel(latitude=slice(LAT_MAX, LAT_MIN)).query(
                    longitude=f"longitude <= {lon_max_gfs} or longitude >= {lon_min_gfs}")
            else:
                ds_cropped = ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(lon_min_gfs, lon_max_gfs))

            filename = WEATHER / f"gfs_med_{gfs_run_date.strftime('%Y%m%d_%H')}_f{hour:02d}.nc"

            ds_cropped.to_netcdf(filename)
            print(f"Saved cropped NetCDF: {filename}")
            return filename

        except Exception as e:
            print(f" [!] Errore o Run {gfs_run_date.strftime('%H:%M')} UTC non pronta. Dettaglio: {e}. Fallback...")
            gfs_run_date = gfs_run_date - timedelta(hours=6)

    raise RuntimeError(f"Impossibile trovare una run GFS valida per l'ora +{hour:02d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the latest GFS 10 m wind forecast for the Mediterranean."
    )
    parser.add_argument(
        "--max-lead-hours",
        type=int,
        default=DEFAULT_MAX_LEAD_HOURS,
        help="Forecast coverage to download from the selected GFS run (default: 72).",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=DEFAULT_INTERVAL_HOURS,
        help="Spacing between forecast fields in hours (default: 3).",
    )
    arguments = parser.parse_args()
    if arguments.max_lead_hours < 0:
        parser.error("--max-lead-hours must be non-negative.")
    if arguments.interval_hours <= 0:
        parser.error("--interval-hours must be positive.")
    if arguments.max_lead_hours % arguments.interval_hours != 0:
        parser.error("--max-lead-hours must be divisible by --interval-hours.")

    forecast_hours = range(
        0,
        arguments.max_lead_hours + arguments.interval_hours,
        arguments.interval_hours,
    )
    WEATHER.mkdir(exist_ok=True, parents=True)
    downloaded = []
    print(
        "Starting Mediterranean GFS download pipeline "
        f"(0 to +{arguments.max_lead_hours} h every {arguments.interval_hours} h).\n"
    )

    for h in forecast_hours:
        try:
            f = download_forecast(h)
            if f:
                downloaded.append(f)
        except Exception as e:
            print(f"[-] Failed to download forecast +{h:02d} h. Reason: {e}")

    print("\n--- Download Task Completed ---")


if __name__ == "__main__":
    main()
