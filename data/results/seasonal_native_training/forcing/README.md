# Seasonal forcing archive

These regional subsets cover seven complete UTC days per season in 2026. Each archive adds 24 hours before the study week and 48 hours after it, yielding 240 hourly samples.

| Block | Core UTC dates | First archived hour | Last archived hour |
|---|---|---|---|
| Winter | Jan 12–18 | Jan 11 00:00 | Jan 20 23:00 |
| Spring | Apr 13–19 | Apr 12 00:00 | Apr 21 23:00 |
| Summer | Jul 13–19 | Jul 12 00:00 | Jul 21 23:00 |
| Autumn | Oct 12–18 | Oct 11 00:00 | Oct 20 23:00 |

The requested crop spans 39.85–40.85° N and 17.65–19.15° E. Each archived grid rounds the bounds outward to its own cell coordinates. Time is the UTC valid hour. The wind is a reanalysis, and the marine fields are archived model series accessed by valid time. The responses do not carry forecast issue times. [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api)

| File | Archived source | NetCDF variables |
|---|---|---|
| `wind.nc` | ERA5 hourly 10 m wind at 0.25° | `u10`, `v10` (m/s) |
| `current.nc` | Météo-France SMOC hourly total surface current at 1/12° | `water_u`, `water_v` (m/s) |
| `wave.nc` | ECMWF WAM hourly wave model at approximately 9 km | `VHM0` (m), `VMDR` (° FROM), `VTPK` (s) |

Wind and current components are eastward and northward. The conversion uses the original API speed and bearing, with meteorological wind FROM and surface current TO. The providers' API values retain their reported precision. SMOC includes Eulerian current, wave drift, and tides. WAM supplies significant wave height, mean wave direction, and spectral peak period. [ERA5 hourly single-level data](https://doi.org/10.24381/cds.adbb2d47) · [Copernicus Marine SMOC product](https://doi.org/10.48670/moi-00016) · [ECMWF wave archive parameters](https://www.ecmwf.int/en/forecasts/datasets/set-ii)

The wave subset assigns the closest reported WAM cell to each position on a regular 0.08° target mesh. Each NetCDF stores the API's actual source latitude and longitude per position. More than one target cell can use the same WAM source cell. The greatest reported source-to-target displacement was 6.59 km. This mapping does not increase the native data resolution.

Marine masks are represented by missing values. Open-Meteo rejected some wave requests containing unavailable grid cells. The acquisition script bisected those requests to isolate 70 locations during the winter block and applies that documented location mask to the same grid for the later seasons. The overlapping wet grid contains 224 complete wave target points representing 165 native WAM points. SMOC returns missing values for 61 of its 280 grid points. Raw JSON and rejection responses are retained under `forcing_raw/`. Each block's `acquisition.json` records URLs, source products, grids, time coverage, and hashes.

## Reproduction

From the V2 root, use a Python environment with NumPy, requests, xarray, and netCDF4. The normalization script resumes existing raw responses. It paces network requests and waits on HTTP 429 responses. Run the winter block first when building a new dataset so its location mask is available for the other blocks.

```sh
python -B data/results/seasonal_native_training/reproduce/acquire_seasonal_forcing.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -B data/results/seasonal_native_training/reproduce/verify_acquired_forcing.py
```

The verifier binds saved NetCDFs and raw responses to the acquisition hashes, checks all registered hourly timestamps, and samples the simulator's seasonal environmental loader at registered GPX points, leg quartiles, and GEBCO-wet locations on 12.5 km and 25 km rings around the two route families. It checks every archived hour and each half-hour midpoint and records the observed wind, current, and wave ranges.
