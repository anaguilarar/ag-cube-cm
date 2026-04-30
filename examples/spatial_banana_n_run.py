"""
End-to-end test for BananaModel using real Guadalupe datacubes.

Runs the Banana-N model over every weather pixel (19x19 = 361 pixels),
collects the full weekly biomass history per pixel, and exports a
spatiotemporal datacube:

    output_banana_biomass.nc
        dims : (week, y, x)
        vars : Avg_Biomass_g_mat, Avg_Fruit_g_mat, Avg_SMN_kg_ha, Yield_kg_ha

Run from the project root:
    python test_banana_real_data.py
"""

import sys
import logging
import traceback
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
from ag_cube_cm.config.loader import load_config
from ag_cube_cm.config.schemas import SimulationConfig
from ag_cube_cm.models.banana_n.base import BananaModel

logging.basicConfig(
    level=logging.WARNING,          # suppress INFO noise during parallel runs
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOIL_PATH    = "data_test/soilgrids_guad.nc"
WEATHER_PATH = "data_test/weather_guadalupe_2021_2023.nc"
OUTPUT_PATH  = "output_banana_biomass.nc"

# ---------------------------------------------------------------------------
# Config — planting date must be within the 2021-2023 weather window
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Per-pixel worker
# ---------------------------------------------------------------------------

def run_pixel(args):
    """Run the full pipeline for one (y, x) pixel. Returns list of weekly dicts."""
    y, x, weather_ds, soil_ds, cfg = args
    try:
        w_slice = weather_ds.sel(y=y, x=x, method="nearest")
        s_slice = soil_ds.sel(y=y, x=x, method="nearest")

        model = BananaModel(cfg)
        model.setup_working_directory(f"{y:.4f}_{x:.4f}")
        try:
            model.prepare_inputs(w_slice, s_slice)
            model.run_simulation()
            records = [
                {
                    "y": y, "x": x,
                    "week":               row["Week"],
                    "Avg_Biomass_g_mat":  row.get("Avg_Bioamass_g_mat", np.nan),
                    "Avg_Fruit_g_mat":    row.get("Avg_Fruit_g_mat",    np.nan),
                    "Avg_SMN_kg_ha":      row.get("Avg_SMN_kg_ha",      np.nan),
                }
                for row in model.history
            ]
            return records
        finally:
            model.cleanup_working_directory()
    except Exception as exc:
        logger.warning("Pixel (%.4f, %.4f) failed: %s", y, x, exc)
        return []

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n=== Banana-N real-data test (Guadalupe) ===\n")

    parser = argparse.ArgumentParser(description="Banana-N real-data test")
    parser.add_argument("--config", default="options/test_banana.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # 1. Load config and datacubes lazily
    print("Loading config and datacubes...")
    cfg        = load_config(args.config)
    weather_ds = xr.open_dataset(cfg.SPATIAL_INFO.weather_path)
    soil_ds    = xr.open_dataset(cfg.SPATIAL_INFO.soil_path)

    print(f"  Weather : {dict(weather_ds.dims)}  vars={list(weather_ds.data_vars)}")
    print(f"  Soil    : {dict(soil_ds.dims)}  vars={list(soil_ds.data_vars)}")

    # 2. Build pixel list from weather grid (lower resolution — 19x19)
    pixels = [
        (float(y), float(x), weather_ds, soil_ds, cfg)
        for y in weather_ds.y.values
        for x in weather_ds.x.values
    ]
    print(f"\nPixels to simulate: {len(pixels)}")

    # 3. Run in parallel (threads — pure Python model is GIL-friendly enough
    #    for I/O overlap during soil/weather slicing)
    ncores = 6
    all_records = []
    failed = 0

    with tqdm(total=len(pixels), desc="Simulating pixels", unit="px") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
            futures = {pool.submit(run_pixel, args): args[:2] for args in pixels}
            for future in concurrent.futures.as_completed(futures):
                records = future.result()
                if records:
                    all_records.extend(records)
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nCompleted: {len(pixels) - failed}/{len(pixels)} pixels succeeded")

    if not all_records:
        print("No results — check logs above for errors.")
        sys.exit(1)

    # 4. Build output datacube (week, y, x)
    print("\nBuilding output datacube...")
    df = pd.DataFrame(all_records)
    print(f"  Records: {len(df)}  weeks={df['week'].nunique()}  pixels={df[['y','x']].drop_duplicates().shape[0]}")

    df_idx = df.set_index(["week", "y", "x"])
    ds_out = xr.Dataset.from_dataframe(df_idx)
    ds_out = ds_out.sortby(["week", "y", "x"])

    # --- CF conventions: coordinate attributes so QGIS/ncview can place the
    #     grid correctly as WGS84 geographic coordinates. ---
    ds_out["y"].attrs.update({
        "standard_name": "latitude",
        "long_name":     "latitude",
        "units":         "degrees_north",
        "axis":          "Y",
    })
    ds_out["x"].attrs.update({
        "standard_name": "longitude",
        "long_name":     "longitude",
        "units":         "degrees_east",
        "axis":          "X",
    })
    ds_out["week"].attrs.update({
        "long_name": "week index (0 = first week after planting)",
        "units":     "weeks",
        "axis":      "T",
    })

    # Add a CRS grid-mapping variable (WGS84 / EPSG:4326)
    import numpy as np
    crs_var = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name":          "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis":             6378137.0,
            "inverse_flattening":          298.257223563,
            "crs_wkt": (
                'GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433],'
                'AUTHORITY["EPSG","4326"]]'
            ),
            "spatial_ref": "EPSG:4326",
        },
    )
    ds_out = ds_out.assign(crs=crs_var)
    for var in ["Avg_Biomass_g_mat", "Avg_Fruit_g_mat", "Avg_SMN_kg_ha"]:
        ds_out[var].attrs["grid_mapping"] = "crs"

    ds_out.attrs.update({
        "description":    "Banana-N weekly biomass simulation - Guadalupe 2021",
        "planting_date":  "2021-03-01",
        "source_weather": str(cfg.SPATIAL_INFO.weather_path),
        "source_soil":    str(cfg.SPATIAL_INFO.soil_path),
        "Conventions":    "CF-1.8",
        "crs":            "EPSG:4326",
    })

    print(ds_out)

    # 5. Close input datasets before writing output (releases file locks)
    weather_ds.close()
    soil_ds.close()

    # 6. Export — overwrite any existing file
    import os as _os
    if _os.path.exists(OUTPUT_PATH):
        _os.remove(OUTPUT_PATH)
    ds_out.to_netcdf(OUTPUT_PATH)
    print(f"\nSaved -> {OUTPUT_PATH}")

    # 7. Quick sanity check — open and close in a context manager
    with xr.open_dataset(OUTPUT_PATH) as ds_check:
        biomass = ds_check["Avg_Biomass_g_mat"]
        valid   = float(biomass.count())
        print(f"\nSanity check:")
        print(f"  Biomass shape : {biomass.dims} {biomass.shape}")
        print(f"  Valid cells   : {int(valid)}")
        print(f"  Mean (week 52): {float(biomass.sel(week=51, method='nearest').mean()):.1f} g/mat")
        print(f"  Max  (week 52): {float(biomass.sel(week=51, method='nearest').max()):.1f} g/mat")
        print(f"\n  CRS check:")
        print(f"    y attrs : {ds_check['y'].attrs}")
        print(f"    x attrs : {ds_check['x'].attrs}")
        print(f"    crs var : {ds_check['crs'].attrs.get('spatial_ref', 'missing')}")
