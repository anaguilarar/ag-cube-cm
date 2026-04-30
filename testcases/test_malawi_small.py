"""
Small Malawi smoke-test: 3x3 pixels, 3 planting windows.
Run from the project root:
    python testcases/test_malawi_small.py
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ag_cube_cm.config.loader import load_config
from ag_cube_cm.models.dssat.base import DSSATModel

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WEATHER_PATH = "D:/Google/My Drive/CIAT/agwise/weather_mwi_2000_2019.nc"
SOIL_PATH    = "D:/Google/My Drive/CIAT/agwise/soil_mwi.nc"
CONFIG_PATH  = "testcases/test_dssat_malawi.yaml"
N_PIXELS_SIDE = 3   # crop a 3×3 grid near the centre
N_WINDOWS     = 3   # test 3 planting windows (w00, w01, w02)


def build_planting_dates(cfg, n_windows):
    base = cfg.MANAGEMENT.planting_date
    step = cfg.MANAGEMENT.planting_window_days
    return [base + timedelta(days=w * step) for w in range(n_windows)]


def window_config(cfg, pdate):
    mgmt_w = cfg.MANAGEMENT.model_copy(update={"planting_date": pdate})
    return cfg.model_copy(update={"MANAGEMENT": mgmt_w})


def run_pixel_window(args):
    pixel_idx, w_idx, pdate, y, x, weather_ds, soil_ds, cfg = args
    dir_name = f"px{pixel_idx}_w{w_idx:02d}"
    result = {
        "pixel_idx": pixel_idx, "window_idx": w_idx,
        "planting_date": str(pdate), "y": y, "x": x,
        "HWAM": np.nan, "flag": 2, "error": "",
    }
    try:
        w_slice = weather_ds.sel(y=y, x=x, method="nearest")
        s_slice = soil_ds.sel(y=y, x=x, method="nearest")
        if (w_slice.to_dataframe().reset_index().dropna().empty or
                s_slice.to_dataframe().reset_index().dropna().empty):
            result["error"] = "no-data pixel"
            return result

        cfg_w = window_config(cfg, pdate)
        model = DSSATModel(cfg_w)
        model.setup_working_directory(dir_name)
        try:
            model.prepare_inputs(w_slice, s_slice, elevation=0.0)
            model.run_simulation()
            outputs = model.collect_outputs()
            hwam = outputs.get("HWAM", np.nan)
            result["HWAM"]  = float(hwam) if hwam not in (None, "", "-99") else np.nan
            result["flag"]  = 0
        except Exception as e:
            result["flag"]  = 1
            result["error"] = str(e)
            logger.warning("px%d w%02d (%.0f,%.0f) pdate=%s failed: %s",
                           pixel_idx, w_idx, y, x, pdate, e)
        finally:
            model.cleanup_working_directory()
    except Exception as e:
        result["flag"]  = 1
        result["error"] = str(e)
    return result


if __name__ == "__main__":
    print("\n=== Malawi small smoke-test (3x3 pixels, 3 windows) ===\n")

    cfg = load_config(CONFIG_PATH)

    print("Loading datacubes...")
    weather_full = xr.open_dataset(WEATHER_PATH)
    soil_full    = xr.open_dataset(SOIL_PATH)
    print(f"  Weather full: {dict(weather_full.sizes)}")
    print(f"  Soil full   : {dict(soil_full.sizes)}")

    # Crop to a small central patch on the weather grid (coarser, controls pixel count)
    cy = len(weather_full.y) // 2
    cx = len(weather_full.x) // 2
    half = N_PIXELS_SIDE // 2
    weather_ds = weather_full.isel(
        y=slice(cy - half, cy + half + 1),
        x=slice(cx - half, cx + half + 1),
    )
    # Soil uses nearest-neighbour lookup so no need to crop it
    soil_ds = soil_full

    print(f"  Weather crop: {dict(weather_ds.sizes)}  "
          f"y=[{float(weather_ds.y.min()):.0f}, {float(weather_ds.y.max()):.0f}]  "
          f"x=[{float(weather_ds.x.min()):.0f}, {float(weather_ds.x.max()):.0f}]")

    planting_dates = build_planting_dates(cfg, N_WINDOWS)
    print(f"\nPlanting windows ({N_WINDOWS}):")
    for i, pd_ in enumerate(planting_dates):
        print(f"  W{i:02d}: {pd_}")

    pixel_coords = {
        idx: (float(y), float(x))
        for idx, (y, x) in enumerate(
            (y, x)
            for y in weather_ds.y.values
            for x in weather_ds.x.values
        )
    }
    n_pixels = len(pixel_coords)
    n_jobs   = n_pixels * N_WINDOWS
    print(f"\nPixels: {n_pixels}  x  Windows: {N_WINDOWS}  =  {n_jobs} jobs")

    jobs = [
        (idx, w_idx, pdate,
         pixel_coords[idx][0], pixel_coords[idx][1],
         weather_ds, soil_ds, cfg)
        for idx in pixel_coords
        for w_idx, pdate in enumerate(planting_dates)
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_pixel_window, j): j[0:2] for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            status = "ok" if r["flag"] == 0 else ("skip" if r["flag"] == 2 else "FAIL")
            hwam   = f"HWAM={r['HWAM']:.0f}" if r["flag"] == 0 and not np.isnan(r["HWAM"]) else ""
            print(f"  px{r['pixel_idx']:02d} w{r['window_idx']:02d} "
                  f"pdate={r['planting_date']}  [{status}]  {hwam}  {r['error'][:80]}")

    weather_full.close()
    soil_full.close()

    df = pd.DataFrame(results)
    ok     = df[df["flag"] == 0]
    skip   = df[df["flag"] == 2]
    failed = df[df["flag"] == 1]

    print(f"\n{'='*55}")
    print(f"ok={len(ok)}  skip={len(skip)}  failed={len(failed)}")
    if len(ok):
        print(f"Mean HWAM: {ok['HWAM'].dropna().mean():.0f} kg/ha")
    if len(failed):
        print("\nFailures:")
        for _, row in failed.iterrows():
            print(f"  px{int(row['pixel_idx'])} w{int(row['window_idx'])} "
                  f"({row['y']:.0f},{row['x']:.0f}): {row['error']}")
    print(f"{'='*55}\n")
