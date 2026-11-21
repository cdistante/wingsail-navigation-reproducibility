# Data provenance and verification

The repository includes the environmental fields used in the archived Otranto layline and seasonal Brindisi studies. The files below support simulation and data audit without a forecast-service account. The matching outcome and artifact records cover additional experiments.

## Otranto regional forecast archive

The files in `data/weather/`, `data/currents/`, and `data/waves/` contain the exact field arrays loaded by the Otranto regional replay. The geographic crop is 39.5–41° N, 18–19.5° E. Wind files retain their original GFS cycle and forecast-hour filenames. The current and wave subset files combine the archive slices selected by the simulator at each valid time, so a fresh checkout preserves the same overlap resolution independently of file checkout dates. The native values and masks are retained.

| Input | Archive | Variables |
|---|---|---|
| Wind, 83 valid times | NOAA/NCEP 0.25° Global Forecast System | 10 m eastward and northward wind velocity |
| Current, 73 timestamps | HYCOM 1/25° Global ESPC-D-V02, surface | Eastward and northward water velocity |
| Waves, 227 timestamps | Copernicus Marine Mediterranean Sea Waves Analysis and Forecast, product MEDSEA_ANALYSISFORECAST_WAV_006_017 | Significant wave height, mean wave direction, and spectral peak period |

The GFS source is the NOAA/NCEP Global Forecast System. The original NetCDF files were made from the 0.25° GFS GRIB2 fields. The crop retains native coordinates and forecast metadata, with local working-file paths omitted. [NOAA/NCEI GFS catalogue](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)

The HYCOM surface series uses the Global ESPC-D-V02 analysis/forecast archive. Support for HYCOM development came from the National Ocean Partnership Program and Office of Naval Research. The U.S. Navy funded data assimilation products, and the DoD High Performance Computing Modernization Program provided computing time. HYCOM lists these data as approved for public release with unlimited distribution. [HYCOM product](https://www.hycom.org/dataserver/espc-d-v02/global-analysis) · [HYCOM data acknowledgement](https://www.hycom.org/publications/acknowledgements/hycom-data)

Credit for the Mediterranean wave and SMOC data: European Union, Copernicus Marine Service Information or Data (2026), © Mercator Ocean. Generated using E.U. Copernicus Marine Service Information. [Product DOI](https://doi.org/10.48670/mds-00373) · [Copernicus Marine licence and citation guidance](https://help.marine.copernicus.eu/en/articles/4444611-citing-copernicus-marine-products-and-services)

The published [field fingerprint record](../data/results/otranto_layline_eta/archive_field_fingerprints.json) registers every loaded variable and coordinate. The distributed regional fields matched those per-variable shapes, data types and SHA-256 hashes after being reloaded with the same providers.

With the [simulator Python environment](RUN_SIMULATION.md), rerun this comparison from the checkout root. The command reads the released receipt.

```sh
python -B data/results/otranto_layline_eta/reproduce/fingerprint_archive.py --check
```

## Seasonal forcing for Brindisi and Otranto

The [forcing archive](../data/results/seasonal_native_training/forcing/README.md) contains 240 hourly samples per seasonal block, with one core week each in January, April, July, and October 2026. It covers Brindisi and Otranto, with a requested crop of 39.85–40.85° N, 17.65–19.15° E and grid-specific outward rounding. The per-block acquisition receipts bind the normalized NetCDF files and the raw responses under `data/results/seasonal_native_training/forcing_raw/` to SHA-256 digests.

Open-Meteo distributes the API data under CC BY 4.0. The study converted wind and current speed and bearing to east and north components, retained the provider masks, and mapped the closest WAM source points to a regular 0.08° target grid. The NetCDF files record native source coordinates to distinguish the target mesh from native resolution. [Open-Meteo terms](https://open-meteo.com/en/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · [Marine API](https://open-meteo.com/en/docs/marine-weather-api)

| Field | Upstream attribution |
|---|---|
| ERA5 10 m wind | Contains modified Copernicus Climate Change Service information (2026), via Open-Meteo. The European Commission and ECMWF bear no responsibility for uses of the information. [ERA5 hourly single-level data](https://doi.org/10.24381/cds.adbb2d47) · [Copernicus Products licence](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) |
| SMOC total surface current | Copernicus Marine SMOC data via Open-Meteo, credited above along with the Mediterranean wave product. [Copernicus Marine SMOC](https://doi.org/10.48670/moi-00016) |
| WAM waves | ECMWF WAM wave fields via Open-Meteo. [ECMWF wave archive](https://www.ecmwf.int/en/forecasts/datasets/set-ii) |

For the acquisition and spatial-sampling audit, use the commands in the [forcing archive documentation](../data/results/seasonal_native_training/forcing/README.md). The saved API responses allow the normalization to run offline. If acquiring data from the API, run winter first so the remaining blocks can reuse its documented WAM location mask. Live forecast updates and later API retrievals can produce different values. Use the saved inputs for the matched comparisons.

## Land and depth

`data/bathymetry/` contains the GEBCO_2026 subset at 37–44° N, 14–22° E in CF NetCDF and GeoTIFF format. The NetCDF SHA-256 is `97af76b5bc06cbfd6cacd8c493c675d2a8e70a69a420f17e89ce152f93dfc410`, as recorded in the Brindisi replay audit. The simulator uses these elevations for depth and land constraints. GEBCO's terms permit redistribution with acknowledgement and advise against using the grid for navigation or safety at sea. [GEBCO Bathymetric Compilation Group (2026), GEBCO_2026 Grid, BODC](https://doi.org/10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa) · [GEBCO terms](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid) · [Regional subset interface](https://betadownload.gebco.net/)

The coastline context layer in `data/coastline/otranto_coastline.geojson` is a geographic crop of Natural Earth's public-domain 1:110m coastline. [Natural Earth terms of use](https://www.naturalearthdata.com/about/terms-of-use/)

## Traffic

The controlled AIS/COLREG cases use synthetic contact trajectories. They combine simulated ship positions and maneuvers with defined report delays and dropout windows. The simulator applies report age to the information available to guidance, and the evaluation uses the synthetic ground truth to check the full uncertainty clearance. The repository includes the traffic and report settings needed by the staged cases.
