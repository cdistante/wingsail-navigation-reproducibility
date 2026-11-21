"""
Download and crop GEBCO bathymetry.

The complete GEBCO archive is downloaded only once.
Afterwards only the Mediterranean/Otranto subset is used.
"""

from html import unescape
from pathlib import Path
from urllib.parse import urljoin
import re
import sys
import zipfile

import requests
from tqdm import tqdm
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import BATHYMETRY

# GEBCO changes its public download URLs from time to time.
# These are tried first because direct URLs are fastest when they work.
GEBCO_URLS = [
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_2025/geotiff/"
        "GEBCO_2025.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_2025/geotiff/"
        "GEBCO_2025_TID.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_2025/geotiff/"
        "GEBCO_2025_SUB_ICE_TOPO.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_2024/geotiff/"
        "GEBCO_2024_TID.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_2024/geotiff/"
        "GEBCO_2024_SUB_ICE_TOPO.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_latest/geotiff/"
        "GEBCO_latest.zip"
    ),
    (
        "https://www.bodc.ac.uk/data/open_download/"
        "gebco/gebco_latest/geo_tiff/"
        "GEBCO_latest.zip"
    ),
]

# Pages that may contain the current public GEBCO ZIP links.
# If the direct URLs above fail, the script searches these pages for suitable ZIP files.
GEBCO_DISCOVERY_PAGES = [
    "https://www.gebco.net/data_and_products/gridded_bathymetry_data/",
    "https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2026/",
    "https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2024/",
    "https://www.bodc.ac.uk/data/open_download/gebco/",
]

LAT_MIN = 30.0
LAT_MAX = 46.0

LON_MIN = -6.0
LON_MAX = 36.0

REQUEST_TIMEOUT_SECONDS = 180


def is_valid_zip(path: Path) -> bool:
    """
    Return True only if path exists and is a readable ZIP archive.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False

    if not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            return bad_member is None
    except zipfile.BadZipFile:
        return False


def response_preview(response: requests.Response, max_chars: int = 500) -> str:
    """
    Return a short preview of a server response.

    This helps diagnose cases where a server returns an HTML error page
    instead of the expected ZIP file.
    """
    try:
        text = response.text.strip()
    except Exception:
        return "<response body could not be decoded>"

    if not text:
        return "<empty response body>"

    return text[:max_chars]


def unique_preserving_order(values: list[str]) -> list[str]:
    """
    Return values with duplicates removed while preserving original order.
    """
    seen = set()
    unique_values = []

    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)

    return unique_values


def looks_like_gebco_geotiff_zip(url: str) -> bool:
    """
    Return True when a URL looks like a GEBCO GeoTIFF ZIP archive.

    GEBCO filenames and folders change over time, so this intentionally checks
    broad but relevant terms instead of one exact filename.
    """
    lowered = url.lower()

    if not lowered.endswith(".zip"):
        return False

    if "gebco" not in lowered:
        return False

    geotiff_terms = [
        "geotiff",
        "geo_tiff",
        "tif",
        "tiff",
    ]

    grid_terms = [
        "sub_ice_topo",
        "tid",
        "grid",
        "gebco_",
        "latest",
    ]

    return any(term in lowered for term in geotiff_terms) or any(
        term in lowered for term in grid_terms
    )


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    """
    Extract absolute links from HTML.

    This intentionally uses the standard library plus regex to avoid adding
    another dependency just for link discovery.
    """
    links = []

    patterns = [
        r"""href\s*=\s*["']([^"']+)["']""",
        r"""src\s*=\s*["']([^"']+)["']""",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            raw_link = unescape(match.group(1).strip())

            if not raw_link:
                continue

            absolute_link = urljoin(base_url, raw_link)
            links.append(absolute_link)

    return unique_preserving_order(links)


def discover_gebco_urls() -> list[str]:
    """
    Search GEBCO/BODC pages for current GEBCO ZIP archive links.

    This makes the downloader more resilient when GEBCO changes its public paths.
    """
    discovered_urls = []

    print("Searching GEBCO pages for current ZIP download links...")

    for page_url in GEBCO_DISCOVERY_PAGES:
        print(f"Checking: {page_url}")

        try:
            response = requests.get(
                page_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Could not read discovery page: {exc}")
            continue

        content_type = response.headers.get("Content-Type", "").lower()

        if "html" not in content_type and "text" not in content_type:
            continue

        page_links = extract_links_from_html(response.text, response.url)

        for link in page_links:
            if looks_like_gebco_geotiff_zip(link):
                discovered_urls.append(link)

    discovered_urls = unique_preserving_order(discovered_urls)

    if discovered_urls:
        print("Discovered possible GEBCO ZIP URLs:")
        for url in discovered_urls:
            print(f"  {url}")
    else:
        print("No GEBCO ZIP URLs were discovered automatically.")

    return discovered_urls


def download_from_url(url: str, destination: Path) -> Path:
    """
    Download one GEBCO archive URL safely.

    The file is first written to a temporary .part file, then atomically moved
    into place only after validation succeeds.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_destination = destination.with_suffix(destination.suffix + ".part")

    if temporary_destination.exists():
        temporary_destination.unlink()

    print("Downloading GEBCO bathymetry archive...")
    print(f"URL:    {url}")
    print(f"Output: {destination}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                preview = response_preview(response)
                raise RuntimeError(
                    "The GEBCO server returned an HTTP error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Final URL: {response.url}\n"
                    f"Server response preview:\n{preview}"
                ) from exc

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "text/plain" in content_type:
                preview = response_preview(response)
                raise RuntimeError(
                    "The GEBCO server returned a text/HTML response instead of a ZIP file.\n"
                    f"Content-Type: {content_type or '<missing>'}\n"
                    f"Response preview:\n{preview}"
                )

            total = int(response.headers.get("content-length", 0))

            with temporary_destination.open("wb") as file, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))

    except requests.RequestException as exc:
        if temporary_destination.exists():
            temporary_destination.unlink()
        raise RuntimeError(f"Network error while downloading GEBCO archive: {exc}") from exc

    if not is_valid_zip(temporary_destination):
        if temporary_destination.exists():
            temporary_destination.unlink()
        raise RuntimeError(
            "Downloaded file is not a valid ZIP archive. "
            "The GEBCO URL may have changed, or the server may have returned an error page."
        )

    temporary_destination.replace(destination)

    print("Download completed and ZIP archive validated.")
    return destination


def download(urls: list[str], destination: Path) -> Path:
    """
    Download the GEBCO archive from the first working URL.

    If all configured URLs fail, discover current GEBCO ZIP links from public
    GEBCO/BODC pages and try those too.

    If an existing archive is invalid, it is removed and downloaded again.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if is_valid_zip(destination):
            print("Valid archive already present.")
            return destination

        print(f"Invalid archive found and removed: {destination}")
        destination.unlink()

    errors = []
    attempted_urls = []

    for url in unique_preserving_order(urls):
        attempted_urls.append(url)

        try:
            return download_from_url(url, destination)
        except RuntimeError as exc:
            errors.append(f"{url}\n{exc}")
            print()
            print("This GEBCO URL failed. Trying the next candidate...")
            print()

    discovered_urls = [
        url for url in discover_gebco_urls()
        if url not in attempted_urls
    ]

    for url in discovered_urls:
        attempted_urls.append(url)

        try:
            return download_from_url(url, destination)
        except RuntimeError as exc:
            errors.append(f"{url}\n{exc}")
            print()
            print("This discovered GEBCO URL failed. Trying the next candidate...")
            print()

    raise RuntimeError(
        "Failed to download GEBCO bathymetry archive from all known and discovered URLs.\n\n"
        "The configured GEBCO URLs returned errors, and automatic discovery did not find "
        "a working ZIP archive.\n\n"
        "Open the GEBCO gridded bathymetry download page in your browser, copy the current "
        "GeoTIFF ZIP URL, and add it to GEBCO_URLS near the top of this script.\n\n"
        "Attempted URLs and errors:\n\n"
        + "\n\n---\n\n".join(errors)
    )


def find_tif_file(directory: Path) -> Path:
    """
    Find the extracted GEBCO GeoTIFF.

    GEBCO archive contents can change over time, so this avoids relying on
    exactly one hard-coded internal filename.
    """
    preferred_names = [
        "GEBCO_latest.tif",
        "GEBCO_2025.tif",
        "GEBCO_2025_TID.tif",
        "GEBCO_2025_SUB_ICE_TOPO.tif",
        "GEBCO_2024_TID.tif",
        "GEBCO_2024_SUB_ICE_TOPO.tif",
    ]

    for name in preferred_names:
        matches = sorted(directory.rglob(name))
        if matches:
            return matches[0]

    tif_files = sorted(directory.rglob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(
            f"No .tif file was found after extracting the archive into: {directory}"
        )

    return tif_files[0]


def extract(zip_file: Path) -> Path:
    """
    Extract the GEBCO ZIP archive and return the GeoTIFF path.
    """
    existing_tif_candidates = [
        BATHYMETRY / "GEBCO_latest.tif",
        BATHYMETRY / "GEBCO_2025.tif",
        BATHYMETRY / "GEBCO_2025_TID.tif",
        BATHYMETRY / "GEBCO_2025_SUB_ICE_TOPO.tif",
        BATHYMETRY / "GEBCO_2024_TID.tif",
        BATHYMETRY / "GEBCO_2024_SUB_ICE_TOPO.tif",
    ]

    for existing_tif in existing_tif_candidates:
        if existing_tif.exists():
            print("Extracted GeoTIFF already present.")
            return existing_tif

    existing_tifs = sorted(BATHYMETRY.rglob("*.tif"))

    if existing_tifs:
        print("Extracted GeoTIFF already present.")
        return existing_tifs[0]

    if not is_valid_zip(zip_file):
        raise RuntimeError(
            f"The archive is not a valid ZIP file: {zip_file}\n"
            "Delete it and run the script again, or let the download step recreate it."
        )

    print("Extracting GEBCO archive...")

    with zipfile.ZipFile(zip_file) as archive:
        archive.extractall(BATHYMETRY)

    extracted_tif = find_tif_file(BATHYMETRY)

    print(f"Extracted GeoTIFF: {extracted_tif}")
    return extracted_tif


def crop(tif: Path) -> Path:
    """
    Crop the GEBCO GeoTIFF to the configured geographic bounds.
    """
    output = BATHYMETRY / "gebco_otranto.tif"

    if output.exists():
        print("Subset already available.")
        return output

    print("Cropping bathymetry subset...")

    with rasterio.open(tif) as src:
        window = from_bounds(
            LON_MIN,
            LAT_MIN,
            LON_MAX,
            LAT_MAX,
            src.transform,
        )

        transform = src.window_transform(window)
        image = src.read(1, window=window)

        profile = src.profile
        profile.update(
            width=image.shape[1],
            height=image.shape[0],
            transform=transform,
        )

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(image, 1)

    print(f"Saved subset: {output}")
    return output


def main() -> None:
    BATHYMETRY.mkdir(exist_ok=True, parents=True)

    zip_file = BATHYMETRY / "GEBCO_bathymetry.zip"

    download(GEBCO_URLS, zip_file)

    tif = extract(zip_file)

    crop(tif)


if __name__ == "__main__":
    main()