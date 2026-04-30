"""
End-to-end DSSAT simulation test using real Guadalupe datacubes.

Bootstraps DSSAT from the bundled static/ folder into D:/tmp/DSSAT048,
then runs the binary for every combination of (land pixel, planting window).

Parallelization strategy
------------------------
All (pixel, window) jobs are submitted as a flat list to a single
ThreadPoolExecutor. This maximises hardware utilisation: instead of
sequencing N windows inside each pixel worker, all pixel×window
combinations run concurrently up to `ncores` threads.

Output
------
    output_dssat_yield.nc
        dims  : (planting_window, y, x)
        vars  : HWAM        — mean yield across simulated years [kg ha⁻¹]
                HWAM_yearly — per-year yield array per pixel×window
                flag        — 0=ok, 1=sim_failed, 2=ocean/no-data

Run from the project root:
    python test_dssat_run_simulation.py
"""

import sys
import os
import logging
import concurrent.futures
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ag_cube_cm.config.schemas import (
    CropConfig,
    FertilizerApplication,
    GeneralInfoConfig,
    ManagementConfig,
    SimulationConfig,
    SpatialInfoConfig,
)
from ag_cube_cm.models.dssat.base import DSSATModel

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOIL_PATH    = "data_test/soilgrids_guad.nc"
WEATHER_PATH = "data_test/weather_guadalupe_2021_2023.nc"
DEM_PATH     = "data_test/guadalupe_dem.tif"
OUTPUT_PATH  = "output_dssat_yield.nc"
DSSAT_TMP    = "D:/tmp"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    return SimulationConfig(
        GENERAL_INFO=GeneralInfoConfig(
            country="Guadeloupe",
            country_code="GLP",
            model="dssat",
            working_path="D:/tmp/dssat_runs",
            dssat_path=DSSAT_TMP,
            ncores=8,
        ),
        SPATIAL_INFO=SpatialInfoConfig(
            feature_name="shapeName",
            soil_path=SOIL_PATH,
            weather_path=WEATHER_PATH,
            dem_path=DEM_PATH,
        ),
        CROP=CropConfig(name="Maize", cultivar="IB1072"),
        MANAGEMENT=ManagementConfig(
            planting_date="2021-03-01",
            n_planting_windows=4,       # simulate 4 different planting dates
            planting_window_days=7,     # each window is 7 days later than the previous
            fertilizer_schedule=[
                FertilizerApplication(days_after_planting=5,   n_kg_ha=50.0, p_kg_ha=30.0),
                FertilizerApplication(days_after_planting=30,  n_kg_ha=100.0),
                FertilizerApplication(days_after_planting=60,  n_kg_ha=50.0),
            ],
        ),
    )

# ---------------------------------------------------------------------------
# DEM sampling
# ---------------------------------------------------------------------------

def sample_dem_elevations(dem_path: str, lats: list, lons: list) -> dict:
    elev_map: dict = {}
    with rasterio.open(dem_path) as src:
        band   = src.read(1)
        nodata = src.nodata
        for lat, lon in zip(lats, lons):
            try:
                row, col = rowcol(src.transform, lon, lat)
                val = band[row, col]
                elev_map[(lat, lon)] = -99.0 if (nodata is not None and val == nodata) else float(val)
            except Exception:
                elev_map[(lat, lon)] = -99.0
    return elev_map

# ---------------------------------------------------------------------------
# Planting-window helpers
# ---------------------------------------------------------------------------

def build_planting_dates(cfg: SimulationConfig) -> list:
    """Return the list of planting dates for all requested windows."""
    base = cfg.MANAGEMENT.planting_date          # datetime.date
    n    = cfg.MANAGEMENT.n_planting_windows or 1
    step = cfg.MANAGEMENT.planting_window_days
    return [base + timedelta(days=w * step) for w in range(n)]


def window_config(cfg: SimulationConfig, pdate) -> SimulationConfig:
    """Return a config copy with a specific planting date for one window."""
    mgmt_w = cfg.MANAGEMENT.model_copy(update={"planting_date": pdate})
    return cfg.model_copy(update={"MANAGEMENT": mgmt_w})

# ---------------------------------------------------------------------------
# Per-(pixel, window) worker
# ---------------------------------------------------------------------------

def run_pixel_window(args):
    """
    Simulate one (pixel, planting-window) combination.

    Directory name px{pixel_idx}_w{window_idx:02d} keeps paths short enough
    for DSSAT's internal limits while uniquely identifying each job.
    """
    pixel_idx, w_idx, pdate, y, x, weather_ds, soil_ds, cfg, elev = args

    dir_name = f"px{pixel_idx}_w{w_idx:02d}"
    result = {
        "pixel_idx":    pixel_idx,
        "window_idx":   w_idx,
        "planting_date": str(pdate),
        "y": y, "x": x,
        "HWAM":         np.nan,
        "HWAM_yearly":  [],
        "flag":         2,
        "error":        "",
    }

    try:
        w_slice = weather_ds.sel(y=y, x=x, method="nearest")
        s_slice = soil_ds.sel(y=y, x=x, method="nearest")

        if (w_slice.to_dataframe().reset_index().dropna().empty or
                s_slice.to_dataframe().reset_index().dropna().empty):
            result["flag"]  = 2
            result["error"] = "ocean/no-data pixel"
            return result

        cfg_w = window_config(cfg, pdate)
        model = DSSATModel(cfg_w)
        model.setup_working_directory(dir_name)
        try:
            model.prepare_inputs(w_slice, s_slice, elevation=elev)
            model.run_simulation()
            outputs = model.collect_outputs()

            hwam = outputs.get("HWAM", np.nan)
            result["HWAM"]        = float(hwam) if hwam not in (None, "", "-99") else np.nan
            result["HWAM_yearly"] = outputs.get("HWAM_yearly", [result["HWAM"]])
            result["flag"]        = 0
        except Exception as e:
            result["flag"]  = 1
            result["error"] = str(e)
            logger.warning("px%d w%02d (%.4f,%.4f) pdate=%s failed: %s",
                           pixel_idx, w_idx, y, x, pdate, e)
        finally:
            model.cleanup_working_directory()

    except Exception as e:
        result["flag"]  = 1
        result["error"] = str(e)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n=== DSSAT real-data simulation — Guadalupe (multi-window) ===\n")

    print("Loading datacubes...")
    weather_ds = xr.open_dataset(WEATHER_PATH)
    soil_ds    = xr.open_dataset(SOIL_PATH)
    cfg        = make_config()

    print(f"  Weather : {dict(weather_ds.sizes)}  vars={list(weather_ds.data_vars)}")
    print(f"  Soil    : {dict(soil_ds.sizes)}  vars={list(soil_ds.data_vars)}")

    # Planting windows
    planting_dates = build_planting_dates(cfg)
    n_windows      = len(planting_dates)
    print(f"\nPlanting windows ({n_windows} total, every "
          f"{cfg.MANAGEMENT.planting_window_days} days):")
    for i, pd_ in enumerate(planting_dates):
        print(f"  W{i:02d}: {pd_}")

    # DEM elevations
    print("\nSampling DEM elevations...")
    ys = [float(y) for y in weather_ds.y.values for _ in weather_ds.x.values]
    xs = [float(x) for _ in weather_ds.y.values for x in weather_ds.x.values]
    elev_map = sample_dem_elevations(DEM_PATH, ys, xs)

    # Pixel index → (y, x)
    pixel_coords: dict = {}
    for idx, (y, x) in enumerate(
        (float(y), float(x))
        for y in weather_ds.y.values
        for x in weather_ds.x.values
    ):
        pixel_coords[idx] = (y, x)

    n_pixels = len(pixel_coords)
    n_jobs   = n_pixels * n_windows
    ncores   = cfg.GENERAL_INFO.ncores

    print(f"\nPixels: {n_pixels}  ×  Windows: {n_windows}  =  {n_jobs} jobs")
    print(f"Threads: {ncores}")
    print(f"DSSAT_HOME: {DSSAT_TMP}/DSSAT048/\n")

    # Build flat job list — all (pixel, window) combinations
    jobs = [
        (idx, w_idx, pdate,
         pixel_coords[idx][0], pixel_coords[idx][1],
         weather_ds, soil_ds, cfg,
         elev_map[pixel_coords[idx]])
        for idx in pixel_coords
        for w_idx, pdate in enumerate(planting_dates)
    ]

    results = []
    with tqdm(total=n_jobs, desc="Running DSSAT", unit="job") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
            futures = {pool.submit(run_pixel_window, j): j[0:2] for j in jobs}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    weather_ds.close()
    soil_ds.close()

    df = pd.DataFrame(results)
    ok     = df[df["flag"] == 0]
    skip   = df[df["flag"] == 2]
    failed = df[df["flag"] == 1]

    print(f"\n{'='*60}")
    print(f"Results:  ok={len(ok)}  ocean/skip={len(skip)}  failed={len(failed)}")
    print(f"{'='*60}")

    if len(ok) > 0:
        print(f"\nYield (HWAM kg/ha) across all land pixels and windows:")
        valid = ok["HWAM"].dropna()
        print(f"  Count : {len(valid)}")
        print(f"  Min   : {valid.min():.0f}")
        print(f"  Mean  : {valid.mean():.0f}")
        print(f"  Max   : {valid.max():.0f}")

        print(f"\nPer-window summary:")
        for w_idx, pdate in enumerate(planting_dates):
            w_ok = ok[ok["window_idx"] == w_idx]["HWAM"].dropna()
            if len(w_ok):
                print(f"  W{w_idx:02d} ({pdate}): n={len(w_ok)}  "
                      f"mean={w_ok.mean():.0f}  min={w_ok.min():.0f}  max={w_ok.max():.0f} kg/ha")

    if len(failed) > 0:
        print(f"\nFailed jobs (first 5):")
        for _, row in failed.head(5).iterrows():
            print(f"  px{int(row['pixel_idx'])} w{int(row['window_idx']):02d} "
                  f"({row['y']:.4f},{row['x']:.4f}) pdate={row['planting_date']}: {row['error']}")

    # -----------------------------------------------------------------------
    # Build output datacube (planting_window, y, x)
    # -----------------------------------------------------------------------
    print("\nBuilding output datacube (planting_window, y, x)...")

    y_vals = sorted(df["y"].unique())
    x_vals = sorted(df["x"].unique())
    yi_map = {v: i for i, v in enumerate(y_vals)}
    xi_map = {v: i for i, v in enumerate(x_vals)}

    hwam_grid = np.full((n_windows, len(y_vals), len(x_vals)), np.nan, dtype=np.float32)
    flag_grid = np.full((n_windows, len(y_vals), len(x_vals)), 2,       dtype=np.int8)

    for _, row in df.iterrows():
        wi = int(row["window_idx"])
        yi = yi_map[row["y"]]
        xi = xi_map[row["x"]]
        hwam_grid[wi, yi, xi] = row["HWAM"]
        flag_grid[wi, yi, xi] = int(row["flag"])

    pdate_strs = [str(pd_) for pd_ in planting_dates]

    ds_out = xr.Dataset(
        {
            "HWAM": (
                ["planting_window", "y", "x"], hwam_grid,
                {"long_name": "Mean grain yield at maturity across simulated years",
                 "units": "kg/ha", "grid_mapping": "crs", "_FillValue": np.nan},
            ),
            "flag": (
                ["planting_window", "y", "x"], flag_grid,
                {"long_name": "Pixel status (0=ok, 1=sim_failed, 2=no_data)",
                 "grid_mapping": "crs"},
            ),
        },
        coords={
            "planting_window": (
                ["planting_window"], np.arange(n_windows),
                {"long_name": "Planting-window index (0-based)"},
            ),
            "planting_date": (
                ["planting_window"], pdate_strs,
                {"long_name": "Calendar planting date for each window"},
            ),
            "y": (["y"], y_vals,
                  {"standard_name": "latitude",  "units": "degrees_north", "axis": "Y"}),
            "x": (["x"], x_vals,
                  {"standard_name": "longitude", "units": "degrees_east",  "axis": "X"}),
        },
    )

    crs_var = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "spatial_ref": "EPSG:4326",
        },
    )
    ds_out = ds_out.assign(crs=crs_var)
    ds_out.attrs.update({
        "description":      "DSSAT maize yield — Guadalupe multi-window simulation",
        "base_planting_date": str(planting_dates[0]),
        "n_planting_windows": n_windows,
        "window_interval_days": cfg.MANAGEMENT.planting_window_days,
        "source_weather":   WEATHER_PATH,
        "source_soil":      SOIL_PATH,
        "Conventions":      "CF-1.8",
        "crs":              "EPSG:4326",
    })

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    ds_out.to_netcdf(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")
    print(ds_out)

    # Sanity check
    with xr.open_dataset(OUTPUT_PATH) as ds_check:
        hwam = ds_check["HWAM"]
        print(f"\nSanity check:")
        print(f"  Shape       : {hwam.dims} {hwam.shape}")
        print(f"  Valid cells : {int(hwam.count())}")
        if int(hwam.count()) > 0:
            print(f"  Mean yield  : {float(hwam.mean()):.0f} kg/ha")
            print(f"  Planting dates: {list(ds_check['planting_date'].values)}")

    print(f"\n{'='*60}")
    if len(failed) == 0 and len(ok) > 0:
        print("All land-pixel × window jobs completed successfully.")
    elif len(ok) > 0:
        print(f"{len(ok)} jobs succeeded, {len(failed)} failed.")
    else:
        print("No successful simulations — check errors above.")
    print(f"{'='*60}\n")

    import sys as _sys, os as _os
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0 if len(failed) == 0 else 1)
