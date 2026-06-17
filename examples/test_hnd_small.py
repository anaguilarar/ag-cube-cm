"""
Quick end-to-end test - small Honduras area
===========================================
Bbox   : ~0.3 x 0.3 deg in Comayagua valley
Period : 2021 (1 year)
Sources: CHIRPS + CHIRTS + AgERA5  (CDS key required) + SoilGrids
DSSAT  : Maize IB1072, 1 planting window, max 5 pixels
"""

from __future__ import annotations
import json, logging, sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# -- config --------------------------------------------------------------------
BASE_DIR    = Path("D:/ag_cube_test/hnd_small")
BBOX        = [-87.5, 14.2, -87.2, 14.5]   # [xmin, ymin, xmax, ymax]
START_DATE  = "2021-01-01"
END_DATE    = "2021-12-31"
SUFFIX      = "hnd_small"
DSSAT_WORK  = "D:/dssat_test"               # no spaces -- Fortran constraint

CROP        = "Maize"
CULTIVAR    = "IB1072"
PLANTING    = "2021-05-15"
N_WINDOWS   = 1
NCORES      = 2
MAX_PIXELS  = 25  # top 2 rows + left 2 cols are edge NaNs; need row 3+ to find valid pixels

SOIL_VARS   = ["clay", "sand", "silt", "bdod", "cfvo",
               "soc", "phh2o", "wv0010", "wv0033", "wv1500"]
SOIL_DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100"]

# -- derived paths -------------------------------------------------------------
CLIMATE_DIR = BASE_DIR / "climate_raw"
SOIL_DIR    = BASE_DIR / "soil_raw"
WEATHER_NC  = CLIMATE_DIR / f"climate_{SUFFIX}_2021_2021.nc"
SOIL_NC     = BASE_DIR / f"soil_{SUFFIX}.nc"
YIELD_NC    = BASE_DIR / f"yield_{SUFFIX}.nc"
AGEO_CFG    = BASE_DIR / f"aggeodata_{SUFFIX}.yaml"
SIM_CFG     = BASE_DIR / f"dssat_{SUFFIX}.yaml"

BASE_DIR.mkdir(parents=True, exist_ok=True)
CLIMATE_DIR.mkdir(exist_ok=True)
SOIL_DIR.mkdir(exist_ok=True)

# -- [1] aggeodata download config --------------------------------------------
print("\n[1/7] Writing aggeodata config ...")
ageo_cfg = {
    "DATES": {"starting_date": START_DATE, "ending_date": END_DATE},
    "SPATIAL_INFO": {"extent": BBOX},
    "CLIMATE": {
        "variables": {
            "pr":     {"source": "chirps"},
            "tasmax": {"source": "chirts"},
            "tasmin": {"source": "chirts"},
            "rsds":   {"source": "agera5"},
        }
    },
    "SOIL": {"enabled": False},
    "GENERAL": {
        "suffix":             SUFFIX,
        "ncores":             NCORES,
        "task":               "download",
        "reference_variable": "pr",
        "agera5_version":     "2_0",
        "target_crs":         "EPSG:4326",
    },
    "PATHS": {"output_path": str(CLIMATE_DIR)},
}
with open(str(AGEO_CFG), "w") as fh:
    yaml.dump(ageo_cfg, fh, default_flow_style=False, sort_keys=False)
print(f"  -> {AGEO_CFG}")


# -- [2] download climate ------------------------------------------------------
print("\n[2/7] Downloading climate data ...")
chirps_raw = CLIMATE_DIR / f"pr_{SUFFIX}_raw"
if chirps_raw.exists() and any(chirps_raw.rglob("*.nc")):
    print(f"  -> already present, skipping")
else:
    from aggeodata.pipelines.download import run_download
    res = run_download(str(AGEO_CFG))
    print(f"  -> {sum(len(v) for v in res.values())} files downloaded")


# -- [3] build climate datacube ------------------------------------------------
print("\n[3/7] Building climate datacube ...")
if WEATHER_NC.exists():
    print(f"  -> already exists: {WEATHER_NC}")
else:
    from aggeodata.pipelines.datacube import run_datacube
    nc_path = run_datacube(str(AGEO_CFG))
    WEATHER_NC = Path(nc_path)
    print(f"  -> {WEATHER_NC}")

import xarray as xr
with xr.open_dataset(str(WEATHER_NC)) as wds:
    print(f"  vars={list(wds.data_vars)}  dims={dict(wds.dims)}")


# -- [4] download soil ---------------------------------------------------------
print("\n[4/7] Downloading soil (SoilGrids) ...")
tif_files = list(SOIL_DIR.glob("*.tif"))
if tif_files:
    print(f"  -> already downloaded ({len(tif_files)} files)")
else:
    from aggeodata.ingestion.soil import SoilGridsDownloader
    dl = SoilGridsDownloader(
        soil_layers=SOIL_VARS,
        depths=SOIL_DEPTHS,
        output_folder=str(SOIL_DIR),
    )
    downloaded = dl.download(boundaries=BBOX)
    print(f"  -> {len(downloaded)} GeoTIFF files")


# -- [5] build soil datacube ---------------------------------------------------
print("\n[5/7] Building soil datacube ...")
if SOIL_NC.exists():
    print(f"  -> already exists: {SOIL_NC}")
else:
    from aggeodata.transform.soil_cube import SoilDataCubeBuilder
    builder = SoilDataCubeBuilder(
        data_folder=str(SOIL_DIR),
        variables=SOIL_VARS,
        reference_variable="wv1500",
        target_crs="EPSG:4326",
    )
    builder.build_and_save(output_path=str(BASE_DIR), filename=SOIL_NC.name)
    print(f"  -> {SOIL_NC}")

with xr.open_dataset(str(SOIL_NC)) as sds:
    print(f"  vars={list(sds.data_vars)}  dims={dict(sds.dims)}")


# -- [6] DSSAT simulation ------------------------------------------------------
print("\n[6/7] Running DSSAT (Maize, max_pixels=%d) ..." % MAX_PIXELS)

sim_config = {
    "GENERAL_INFO": {
        "country": "Honduras", "country_code": "HND",
        "model": "dssat",
        "working_path": DSSAT_WORK,
        "dssat_path": None,
        "ncores": NCORES,
    },
    "SPATIAL_INFO": {
        "feature_name": "shapeName", "adm_level": 2, "feature": None,
        "weather_path": str(WEATHER_NC),
        "soil_path":    str(SOIL_NC),
        "output_path":  str(YIELD_NC),
        "dem_path":     None,
    },
    "CROP": {"name": CROP, "cultivar": CULTIVAR},
    "MANAGEMENT": {
        "planting_date": PLANTING,
        "n_planting_windows": N_WINDOWS,
        "planting_window_days": 14,
    },
}
with open(str(SIM_CFG), "w") as fh:
    yaml.dump(sim_config, fh, default_flow_style=False, sort_keys=False)

from ag_cube_cm.mcp_server import run_simulation
result = json.loads(run_simulation(config_path=str(SIM_CFG), max_pixels=MAX_PIXELS))

if result["status"] != "ok":
    print(f"\n  [ERROR] {result['message']}")
    sys.exit(1)

print(f"  pixels ok      : {result['pixels_ok']}")
print(f"  pixels skipped : {result['pixels_skipped']}")
print(f"  pixels failed  : {result['pixels_failed']}")
print(f"  years          : {result['years']}")
print(f"  mean HWAM      : {result['mean_hwam_kg_ha']} kg/ha")
print(f"  output         : {result['output_path']}")


# -- [7] quick plot ------------------------------------------------------------
print("\n[7/7] Plotting results ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

yield_ds = xr.open_dataset(result["output_path"])
hwam = yield_ds["HWAM"]

mean_yield = hwam.mean(dim=["planting_window", "year"])
fig, ax = plt.subplots(figsize=(6, 5))
valid_vals = mean_yield.values[~np.isnan(mean_yield.values)]
if valid_vals.size > 0:
    mean_yield.plot(ax=ax, cmap="YlGn", vmin=0,
                    cbar_kwargs={"label": "Mean yield (kg/ha)"})
else:
    ax.text(0.5, 0.5, "No valid pixels", transform=ax.transAxes, ha="center")
ax.set_title(f"Maize {CULTIVAR} -- Honduras test area (2021)")
plt.tight_layout()
out_png = BASE_DIR / "yield_map.png"
plt.savefig(str(out_png), dpi=120, bbox_inches="tight")
plt.close()
print(f"  -> {out_png}")

yield_ds.close()

print("\n--- Test complete ---")
print(f"  Weather : {WEATHER_NC}")
print(f"  Soil    : {SOIL_NC}")
print(f"  Yield   : {YIELD_NC}")
print(f"  Map     : {out_png}")
