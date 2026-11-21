"""
Download Natural Earth coastline
and crop it to the Mediterranean/Otranto area.
"""

from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import COASTLINE


LAT_MIN = 30.0
LAT_MAX = 46.0

LON_MIN = -6.0
LON_MAX = 36.0


def main():
    COASTLINE.mkdir(exist_ok=True, parents=True)

    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/"
        "110m/physical/"
        "ne_110m_coastline.zip"
    )

    bbox_geometry = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

    bbox = gpd.GeoDataFrame(
        geometry=[bbox_geometry],
        crs="EPSG:4326",
    )

    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    elif world.crs != bbox.crs:
        world = world.to_crs(bbox.crs)

    coast = gpd.clip(world, bbox)

    outfile = COASTLINE / "otranto_coastline.geojson"

    coast.to_file(outfile, driver="GeoJSON")

    print(outfile)


if __name__ == "__main__":
    main()