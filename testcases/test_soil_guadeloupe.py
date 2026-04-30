"""
Smoke test — download SoilGrids GeoTIFFs for a small Guadeloupe bbox
and stack them into a multi-depth NetCDF datacube.

Run from the project root:
    python testcases/test_soil_guadeloupe.py

What it checks
--------------
1. SoilGridsDownloader downloads the expected files.
2. SoilDataCubeBuilder can import cleanly (no missing modules).
3. build_and_save() produces a valid NetCDF with a 'depth' dimension.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ag_cube_cm.ingestion.soil import SoilGridsDownloader
from ag_cube_cm.transform.soil_cube import SoilDataCubeBuilder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BBOX = [-61.85, 15.85, -61.05, 16.55]   # [xmin, ymin, xmax, ymax] — Guadeloupe
OUTPUT_FOLDER = "D:/tmp/soil_guad_test"
NC_FILENAME   = "soilgrids_guad_test.nc"

VARIABLES = [
    "clay", "sand", "silt", "bdod", "cfvo",
    "nitrogen", "phh2o", "soc", "wv0010", "wv0033", "wv1500",
]
DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100"]

# ---------------------------------------------------------------------------
# Step 1 — Download raw GeoTIFFs
# ---------------------------------------------------------------------------

print("\n=== Step 1: Download SoilGrids GeoTIFFs ===\n")
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Delete any previously downloaded wv* files — earlier downloads used a
# broken height/width formula that produced 1x1-pixel files.
for stale in Path(OUTPUT_FOLDER).glob("wv*.tif"):
    stale.unlink()
    print(f"  Deleted stale file: {stale.name}")

dl = SoilGridsDownloader(
    soil_layers=VARIABLES,
    depths=DEPTHS,
    output_folder=OUTPUT_FOLDER,
)
downloaded = dl.download(boundaries=BBOX)

print(f"\nFiles downloaded: {len(downloaded)}")
tif_files = list(Path(OUTPUT_FOLDER).rglob("*.tif"))
print(f"TIF files found : {len(tif_files)}")
for f in sorted(tif_files)[:10]:
    print(f"  {f.relative_to(OUTPUT_FOLDER)}")
if len(tif_files) > 10:
    print(f"  ... and {len(tif_files) - 10} more")

if not tif_files:
    print("\nERROR: No TIF files downloaded — check SoilGrids connection and bbox.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 — Stack into multi-depth NetCDF
# ---------------------------------------------------------------------------

print("\n=== Step 2: Build multi-depth NetCDF cube ===\n")

builder = SoilDataCubeBuilder(
    data_folder=OUTPUT_FOLDER,
    variables=VARIABLES,
    # Do NOT pass extent here — the downloaded TIFs are already clipped to
    # the area of interest and are in ESRI:54052 (Homolosine). Passing a
    # WGS84 bbox as the crop window returns no data.
    reference_variable="wv1500",
    target_crs="EPSG:4326",
)

nc_path = builder.build_and_save(
    output_path=OUTPUT_FOLDER,
    filename=NC_FILENAME,
)

print(f"\nSaved -> {nc_path}")

# ---------------------------------------------------------------------------
# Step 3 — Sanity check the output
# ---------------------------------------------------------------------------

print("\n=== Step 3: Sanity check ===\n")

import xarray as xr

with xr.open_dataset(nc_path) as ds:
    print(f"Dimensions : {dict(ds.sizes)}")
    print(f"Variables  : {list(ds.data_vars)}")
    print(f"Depth vals : {list(ds.depth.values) if 'depth' in ds.dims else 'N/A'}")
    # CRS is stored in the spatial_ref variable (CF convention), not in attrs
    crs_str = ds.attrs.get("crs", "")
    if not crs_str and "spatial_ref" in ds:
        crs_str = str(ds["spatial_ref"].attrs.get("crs_wkt", ds["spatial_ref"].attrs.get("spatial_ref", "")))[:60]
    print(f"CRS        : {crs_str or 'not set'}")

    if "depth" not in ds.dims:
        print("\nWARNING: 'depth' dimension missing — cube did not stack correctly.")
    else:
        print(f"\nOK: {len(ds.depth)} depth layers, "
              f"{len(ds.data_vars)} variables, "
              f"shape {dict(ds.sizes)}")

print("\n=== Done ===\n")
