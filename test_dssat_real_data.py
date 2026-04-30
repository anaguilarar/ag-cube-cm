"""
End-to-end input-file generation test for DSSATModel using real Guadalupe datacubes.

Does NOT require the DSSAT binary.  For every land pixel (weather grid, 19x19),
this test:
  1. Samples elevation from the DEM GeoTIFF
  2. Calls prepare_inputs() → writes WTHE0001.WTH, TRAN0001.SOL, EXPS0001.MZX,
     DSSBatch.v48 inside a per-pixel temp directory
  3. Validates each generated file for correct units, header presence, and row count
  4. Collects per-pixel metadata into a summary DataFrame

Run from the project root:
    python test_dssat_real_data.py
"""

import sys
import logging
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray          # noqa: F401  — activates .rio accessor on xr objects
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    return SimulationConfig(
        GENERAL_INFO=GeneralInfoConfig(
            country="Guadeloupe",
            country_code="GLP",
            model="dssat",
            working_path="./tmp_dssat_real",
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
            fertilizer_schedule=[
                FertilizerApplication(days_after_planting=5,   n_kg_ha=50.0, p_kg_ha=30.0),
                FertilizerApplication(days_after_planting=30,  n_kg_ha=100.0),
                FertilizerApplication(days_after_planting=60,  n_kg_ha=50.0),
            ],
        ),
    )

# ---------------------------------------------------------------------------
# DEM sampling helper
# ---------------------------------------------------------------------------

def sample_dem_elevations(dem_path: str, lats: list[float], lons: list[float]) -> dict:
    """Pre-sample elevations for all (lat, lon) pairs and return a dict.

    Opens and closes rasterio in one shot so no file handle leaks into the
    simulation loop or Python's shutdown garbage collector.
    """
    elev_map: dict[tuple[float, float], float] = {}
    with rasterio.open(dem_path) as src:
        band = src.read(1)
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
# Per-pixel worker
# ---------------------------------------------------------------------------

def run_pixel(args):
    y, x, weather_ds, soil_ds, cfg, elev = args
    result = {"y": y, "x": x, "status": "ok", "error": "",
              "wth_rows": 0, "sol_layers": 0, "tmax_max": np.nan,
              "srad_max": np.nan, "elev": elev}
    try:
        w_slice = weather_ds.sel(y=y, x=x, method="nearest")
        s_slice = soil_ds.sel(y=y, x=x, method="nearest")

        # Quick land check
        df_wth = w_slice.to_dataframe().reset_index().dropna()
        df_sol = s_slice.to_dataframe().reset_index().dropna()
        if df_wth.empty or df_sol.empty:
            result["status"] = "skip"
            result["error"] = "ocean/no-data pixel"
            return result

        model = DSSATModel(cfg)
        model.setup_working_directory(f"{y:.4f}_{x:.4f}")
        try:
            model.prepare_inputs(w_slice, s_slice, elevation=elev)

            # --- validate WTH ---
            wth_file = model.working_dir / "WTHE0001.WTH"
            assert wth_file.exists(), "WTH file not created"
            wth_lines = wth_file.read_text().splitlines()
            data_rows = [l for l in wth_lines if l and not l.startswith(("*", "@", "!"))]
            result["wth_rows"] = len(data_rows)
            assert len(data_rows) > 300, f"WTH too short: {len(data_rows)} rows"
            # Check that the converted srad is in reasonable range (MJ m-2 d-1, not J)
            first_data = data_rows[0].split()
            srad_val = float(first_data[1])
            result["srad_max"] = srad_val
            assert srad_val < 100, f"srad looks like J not MJ: {srad_val}"
            # Check temperatures are in Celsius range
            tmax_val = float(first_data[2])
            result["tmax_max"] = tmax_val
            assert tmax_val < 60, f"tmax looks like Kelvin: {tmax_val}"

            # --- validate SOL ---
            sol_file = model.working_dir / "TRAN0001.SOL"
            assert sol_file.exists(), "SOL file not created"
            sol_lines = sol_file.read_text().splitlines()
            layer_rows = [l for l in sol_lines if l and not l.startswith(("*", "@", "!")) and l.strip()]
            result["sol_layers"] = len(layer_rows)
            assert len(layer_rows) >= 3, f"SOL has too few layers: {len(layer_rows)}"

            # --- validate MZX ---
            mzx_file = model.working_dir / "EXPS0001.MZX"
            assert mzx_file.exists(), "MZX file not created"
            mzx_text = mzx_file.read_text()
            assert "PLANTING DETAILS" in mzx_text, "MZX missing planting details"
            assert "21060" in mzx_text, "MZX planting date (21060=2021-03-01) not found"
            assert "FERTILIZERS" in mzx_text, "MZX missing fertilizer section"

            # --- validate batch ---
            batch_file = model.working_dir / "DSSBatch.v48"
            assert batch_file.exists(), "DSSBatch.v48 not created"

        finally:
            model.cleanup_working_directory()

    except AssertionError as e:
        result["status"] = "fail"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n=== DSSAT real-data input-file test (Guadalupe) ===\n")

    print("Loading datacubes...")
    weather_ds = xr.open_dataset(WEATHER_PATH)
    soil_ds    = xr.open_dataset(SOIL_PATH)
    cfg        = make_config()

    print(f"  Weather : {dict(weather_ds.sizes)}  vars={list(weather_ds.data_vars)}")
    print(f"  Soil    : {dict(soil_ds.sizes)}  vars={list(soil_ds.data_vars)}")

    print("\nSampling DEM elevations...")
    ys = [float(y) for y in weather_ds.y.values for _ in weather_ds.x.values]
    xs = [float(x) for _ in weather_ds.y.values for x in weather_ds.x.values]
    with rasterio.open(DEM_PATH) as dem_src:
        print(f"  DEM CRS : {dem_src.crs}  shape={dem_src.shape}")
        print(f"  DEM bounds: {dem_src.bounds}")
    elev_map = sample_dem_elevations(DEM_PATH, ys, xs)  # rasterio closed inside

    pixels = [
        (float(y), float(x), weather_ds, soil_ds, cfg, elev_map[(float(y), float(x))])
        for y in weather_ds.y.values
        for x in weather_ds.x.values
    ]
    print(f"\nPixels to test: {len(pixels)}")

    ncores = 6
    results = []

    with tqdm(total=len(pixels), desc="Generating DSSAT inputs", unit="px") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
            futures = {pool.submit(run_pixel, args): args[:2] for args in pixels}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # Close xarray datasets now — rasterio was already closed inside sample_dem_elevations.
    weather_ds.close()
    soil_ds.close()

    df = pd.DataFrame(results)
    ok     = df[df["status"] == "ok"]
    skip   = df[df["status"] == "skip"]
    fail   = df[df["status"] == "fail"]
    errors = df[df["status"] == "error"]

    print(f"\n{'='*55}")
    print(f"Results:  ok={len(ok)}  skip={len(skip)}  fail={len(fail)}  error={len(errors)}")
    print(f"{'='*55}")

    if len(ok) > 0:
        print(f"\nFile quality (land pixels only):")
        print(f"  WTH rows     : min={ok['wth_rows'].min():.0f}  max={ok['wth_rows'].max():.0f}  mean={ok['wth_rows'].mean():.0f}")
        print(f"  SOL layers   : min={ok['sol_layers'].min():.0f}  max={ok['sol_layers'].max():.0f}")
        print(f"  srad (MJ/d)  : {ok['srad_max'].mean():.2f}  [should be 5-30, not millions]")
        print(f"  tmax (C)     : {ok['tmax_max'].mean():.1f}  [should be ~25-35, not ~300]")
        print(f"  elevation (m): min={ok['elev'].min():.0f}  max={ok['elev'].max():.0f}  mean={ok['elev'].mean():.0f}")

    if len(fail) > 0:
        print(f"\nFailures:")
        for _, row in fail.iterrows():
            print(f"  ({row['y']:.4f}, {row['x']:.4f}): {row['error']}")

    if len(errors) > 0:
        print(f"\nErrors:")
        for _, row in errors.head(5).iterrows():
            print(f"  ({row['y']:.4f}, {row['x']:.4f}): {row['error']}")

    # Show one example WTH snippet
    if len(ok) > 0:
        ex = ok.iloc[0]
        print(f"\nExample pixel ({ex['y']:.4f}, {ex['x']:.4f})  elev={ex['elev']:.0f}m:")
        cfg2 = make_config()
        y2, x2 = float(ex["y"]), float(ex["x"])
        elev2 = float(ex["elev"])
        # Re-open input datasets just for the snippet, close immediately after
        with xr.open_dataset(WEATHER_PATH) as _wds, xr.open_dataset(SOIL_PATH) as _sds:
            w2 = _wds.sel(y=y2, x=x2, method="nearest").load()
            s2 = _sds.sel(y=y2, x=x2, method="nearest").load()
        m2 = DSSATModel(cfg2)
        m2.setup_working_directory("example_pixel")
        m2.prepare_inputs(w2, s2, elevation=elev2)
        wth_text = (m2.working_dir / "WTHE0001.WTH").read_text()
        sol_text = (m2.working_dir / "TRAN0001.SOL").read_text()
        mzx_lines = (m2.working_dir / "EXPS0001.MZX").read_text().splitlines()
        m2.cleanup_working_directory()

        print("\n--- WTHE0001.WTH (first 8 lines) ---")
        for line in wth_text.splitlines()[:8]:
            print(" ", line)

        print("\n--- TRAN0001.SOL (layer section) ---")
        for line in sol_text.splitlines():
            if line.startswith("@  SLB") or (line.strip() and not line.startswith(("*", "@", "!"))):
                print(" ", line)

        print("\n--- EXPS0001.MZX (planting + fertilizer) ---")
        in_section = False
        for line in mzx_lines:
            if "*PLANTING" in line or "*FERTILIZER" in line:
                in_section = True
            if "*SIMULATION" in line:
                in_section = False
            if in_section:
                print(" ", line)

    print(f"\n{'='*55}")
    if len(fail) == 0 and len(errors) == 0 and len(ok) > 0:
        print("All land pixels PASSED input-file validation.")
    else:
        print("Some pixels had issues — check output above.")
    print(f"{'='*55}\n")

    # Flush output buffers, then use os._exit to bypass the GDAL/netCDF4
    # C-level finalizer that fires during Python's shutdown sequence on Windows
    # and produces a spurious "Error in sys.excepthook" on stderr.
    import sys as _sys, os as _os
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0 if (len(fail) == 0 and len(errors) == 0) else 1)
