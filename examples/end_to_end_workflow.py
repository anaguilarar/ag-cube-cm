"""
End-to-end workflow: multi-source climate + soil → spatial yield potential map
===============================================================================

Packages
--------
    pip install aggeodata ag-cube-cm matplotlib

Data sources
------------
  Climate (via aggeodata)
    CHIRPS   v3     — daily precipitation      0.05°  no API key
    CHIRTS-ERA5     — daily Tmax / Tmin         0.05°  no API key
    AgERA5          — daily solar radiation     0.1°   CDS API key required (*)

  Soil (via aggeodata)
    SoilGrids       — physical + hydraulic props 250m / 1km  no API key

(*) CDS API key setup — one-time, free registration at https://cds.climate.copernicus.eu
    Then create the file  ~/.cdsapirc  with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-key-here>

Workflow
--------
  [aggeodata]  run_download    — CHIRPS + CHIRTS + AgERA5 raw files
  [aggeodata]  run_datacube    — stack into climate_<suffix>_YYYY_YYYY.nc
                                 (CF names: pr, tasmax, tasmin, rsds | dim: time)
  [aggeodata]  SoilGridsDownloader + SoilDataCubeBuilder
                               — soil_<suffix>.nc  (dim: depth)
  [ag_cube_cm] run_simulation  — DSSAT per pixel → yield_<suffix>.nc
                                 (HWAM kg/ha | dims: planting_window × year × y × x)

Region  : Honduras — Comayagua valley  (~1° × 1°)
Period  : 2020–2022  (3 growing seasons)
Crop    : Maize, cultivar IB1072
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION — edit these paths and parameters
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("D:/ag_cube_example/hnd_comayagua")
BBOX       = [-87.8, 14.0, -86.8, 15.0]   # [xmin, ymin, xmax, ymax]  WGS-84
START_DATE = "2020-01-01"
END_DATE   = "2022-12-31"
SUFFIX     = "hnd_comayagua"

# DSSAT working directory — path MUST contain no spaces (Fortran limitation)
DSSAT_WORK = "D:/dssat_runs"

# Crop model settings
CROP           = "Maize"
CULTIVAR       = "IB1072"       # tropical maize
PLANTING_DATE  = "2020-05-01"   # base date; year is replaced for each season
N_WINDOWS      = 3              # number of planting-window scenarios
WINDOW_DAYS    = 14             # days between consecutive windows
NCORES         = 4

# Soil variables (standard DSSAT set)
SOIL_VARS   = ["clay", "sand", "silt", "bdod", "cfvo",
               "soc", "phh2o", "wv0010", "wv0033", "wv1500"]
SOIL_DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100"]

# ──────────────────────────────────────────────────────────────────────────────
# Derived paths (do not edit below this line)
# ──────────────────────────────────────────────────────────────────────────────

CLIMATE_DIR = BASE_DIR / "climate_raw"
SOIL_DIR    = BASE_DIR / "soil_raw"

# aggeodata run_datacube saves the datacube here automatically
START_YEAR  = START_DATE[:4]
END_YEAR    = END_DATE[:4]
WEATHER_NC  = CLIMATE_DIR / f"climate_{SUFFIX}_{START_YEAR}_{END_YEAR}.nc"

SOIL_NC     = BASE_DIR / f"soil_{SUFFIX}.nc"
YIELD_NC    = BASE_DIR / f"yield_{SUFFIX}.nc"
AGEO_CONFIG = BASE_DIR / f"aggeodata_{SUFFIX}.yaml"
SIM_CONFIG  = BASE_DIR / f"dssat_{SUFFIX}.yaml"

BASE_DIR.mkdir(parents=True, exist_ok=True)
CLIMATE_DIR.mkdir(exist_ok=True)
SOIL_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Write the aggeodata download + datacube config
# ──────────────────────────────────────────────────────────────────────────────
# A single YAML config drives both the download phase and the datacube assembly
# phase.  The 'task' key is overridden below when calling each phase.

print("\n[1/7] Writing aggeodata config …")

aggeodata_cfg = {
    "DATES": {
        "starting_date": START_DATE,
        "ending_date":   END_DATE,
    },
    "SPATIAL_INFO": {
        # Bounding box: [xmin, ymin, xmax, ymax]
        "extent": BBOX,
    },
    "CLIMATE": {
        "variables": {
            # precipitation — CHIRPS v3 (0.05°, daily, no API key)
            "pr":     {"source": "chirps"},
            # max/min temperature — CHIRTS-ERA5 (0.05°, daily, no API key)
            "tasmax": {"source": "chirts"},
            "tasmin": {"source": "chirts"},
            # solar radiation — AgERA5 (0.1°, daily, CDS API key required)
            "rsds":   {"source": "agera5"},
            # Optional — uncomment to add wind speed from AgERA5:
            # "sfcWind": {"source": "agera5"},
        }
    },
    "SOIL": {"enabled": False},   # soil handled separately below
    "GENERAL": {
        "suffix":             SUFFIX,
        "ncores":             2,          # downloads run in parallel
        "task":               "download", # overridden per phase
        "reference_variable": "pr",       # CHIRPS grid = spatial reference
        "agera5_version":     "2_0",
        "target_crs":         "EPSG:4326",
    },
    "PATHS": {
        "output_path": str(CLIMATE_DIR),
    },
}

with open(str(AGEO_CONFIG), "w") as fh:
    yaml.dump(aggeodata_cfg, fh, default_flow_style=False, sort_keys=False)
print(f"  → {AGEO_CONFIG}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download climate data (CHIRPS + CHIRTS + AgERA5)
# ──────────────────────────────────────────────────────────────────────────────
# run_download saves files to:
#   pr_<suffix>_raw/        YYYY/chirps_pr_YYYYMMDD.nc
#   tasmax_<suffix>_raw/    tmax/YYYY/chirts_tmax_YYYYMMDD.nc
#   tasmin_<suffix>_raw/    tmin/YYYY/chirts_tmin_YYYYMMDD.nc
#   rsds_<suffix>_raw/      YYYY.zip  (AgERA5 yearly archives)

print("\n[2/7] Downloading climate data …")
print("  Sources: CHIRPS (pr) | CHIRTS-ERA5 (tasmax, tasmin) | AgERA5 (rsds)")

# Check whether data is already downloaded (at least CHIRPS)
chirps_raw = CLIMATE_DIR / f"pr_{SUFFIX}_raw"
already_downloaded = chirps_raw.exists() and any(chirps_raw.rglob("*.nc"))

if already_downloaded:
    print(f"  → data already present in {CLIMATE_DIR}, skipping download")
else:
    from aggeodata.pipelines.download import run_download
    download_results = run_download(str(AGEO_CONFIG))
    print(f"  → downloaded {sum(len(v) for v in download_results.values())} files")
    for cf_var, files in download_results.items():
        print(f"     {cf_var:<10} : {len(files)} entries")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build climate datacube
# ──────────────────────────────────────────────────────────────────────────────
# run_datacube:
#   1. Discovers downloaded files for each variable
#   2. Finds common dates across all variables
#   3. Resamples everything to the reference variable (CHIRPS) grid
#   4. Concatenates along the time dimension
#   5. Saves  climate_<suffix>_<ys>_<ye>.nc  with CF names
#
# Output datacube structure:
#   Dimensions : time × y × x
#   Variables  : pr (mm/day) | tasmax (°C) | tasmin (°C) | rsds (J/m²/day)
#
# The ag_cube_cm DSSATModel automatically aliases these CF names:
#   pr → precipitation  |  tasmax → tmax  |  tasmin → tmin  |  rsds → solar_radiation

print("\n[3/7] Building climate datacube …")

if WEATHER_NC.exists():
    print(f"  → already exists: {WEATHER_NC}")
else:
    # Patch the task field to 'datacube' without overwriting the config file
    import xarray as xr
    from aggeodata.config.loader import load_config
    from aggeodata.pipelines.datacube import run_datacube

    nc_path = run_datacube(str(AGEO_CONFIG))
    print(f"  → {nc_path}")
    WEATHER_NC = Path(nc_path)   # update in case suffix/year changed

# Inspect the datacube
import xarray as xr
with xr.open_dataset(str(WEATHER_NC)) as wds:
    print(f"  variables : {list(wds.data_vars)}")
    print(f"  dims      : {dict(wds.dims)}")
    time_vals = wds.coords.get("time", wds.coords.get("date"))
    if time_vals is not None:
        import pandas as pd
        dates = pd.to_datetime(time_vals.values)
        print(f"  period    : {dates[0].date()} → {dates[-1].date()}  ({len(dates)} days)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Download soil data (SoilGrids)
# ──────────────────────────────────────────────────────────────────────────────
# Physical/chemical variables (clay, sand, …) at 250 m via ISRIC WCS API.
# Hydraulic variables (wv0010, wv0033, wv1500) at 1 km via Google Storage.
# All clipped to BBOX and saved as GeoTIFFs.

print("\n[4/7] Downloading soil data (SoilGrids) …")

if list(SOIL_DIR.glob("*.tif")):
    print(f"  → already downloaded ({len(list(SOIL_DIR.glob('*.tif')))} files)")
else:
    from aggeodata.ingestion.soil import SoilGridsDownloader
    soil_dl = SoilGridsDownloader(
        soil_layers=SOIL_VARS,
        depths=SOIL_DEPTHS,
        output_folder=str(SOIL_DIR),
    )
    downloaded = soil_dl.download(boundaries=BBOX)
    print(f"  → {len(downloaded)} GeoTIFF files in {SOIL_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build soil datacube
# ──────────────────────────────────────────────────────────────────────────────
# SoilDataCubeBuilder:
#   1. Loads each GeoTIFF, assigns native CRS (ESRI:54052 or EPSG:4326)
#   2. Reprojects every layer to EPSG:4326
#   3. Co-registers all variables to the reference variable's grid (wv1500 = 1 km)
#   4. Stacks along a 'depth' dimension (e.g. "0-5", "5-15", …)
#   5. Saves as compressed NetCDF
#
# Output datacube structure:
#   Dimensions : depth × y × x
#   Variables  : clay (g/kg) | sand (g/kg) | bdod (cg/cm³) | wv1500 (10⁻³ cm³/cm³) | …
#
# The DSSATModel reads these directly — unit conversion to DSSAT format happens
# inside DSSATModel._write_sol().

print("\n[5/7] Building soil datacube …")

if SOIL_NC.exists():
    print(f"  → already exists: {SOIL_NC}")
else:
    from aggeodata.transform.soil_cube import SoilDataCubeBuilder
    builder = SoilDataCubeBuilder(
        data_folder=str(SOIL_DIR),
        variables=SOIL_VARS,
        reference_variable="wv1500",
        target_crs="EPSG:4326",
    )
    builder.build_and_save(
        output_path=str(BASE_DIR),
        filename=SOIL_NC.name,
    )
    print(f"  → {SOIL_NC}")

with xr.open_dataset(str(SOIL_NC)) as sds:
    print(f"  variables : {list(sds.data_vars)}")
    print(f"  dims      : {dict(sds.dims)}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Write simulation config and run DSSAT (ag_cube_cm)
# ──────────────────────────────────────────────────────────────────────────────
# Variable name translation (CF → DSSAT) is handled automatically inside
# DSSATModel.prepare_inputs() since ag_cube_cm 0.2.0:
#   pr        → precipitation
#   tasmax    → tmax
#   tasmin    → tmin
#   rsds      → solar_radiation   (unit auto-conv: J/m²/d → MJ/m²/d if > 10 000)
#   time dim  → date column
#
# No manual renaming needed.

print("\n[6/7] Configuring and running DSSAT simulation …")

sim_config = {
    "GENERAL_INFO": {
        "country":      "Honduras",
        "country_code": "HND",
        "model":        "dssat",
        "working_path": DSSAT_WORK,   # NO spaces — Fortran path constraint
        "dssat_path":   None,         # None → use the binary bundled with ag_cube_cm
        "ncores":       NCORES,
    },
    "SPATIAL_INFO": {
        "feature_name": "shapeName",
        "adm_level":    2,
        "feature":      None,         # None → use entire bbox
                                      # e.g. "Comayagua" → clip to one district
        "weather_path": str(WEATHER_NC),
        "soil_path":    str(SOIL_NC),
        "output_path":  str(YIELD_NC),
        "dem_path":     None,
    },
    "CROP": {
        "name":     CROP,
        "cultivar": CULTIVAR,
    },
    "MANAGEMENT": {
        "planting_date":        PLANTING_DATE,
        "n_planting_windows":   N_WINDOWS,
        "planting_window_days": WINDOW_DAYS,
        # Rainfed scenario (no fertilizer).
        # Uncomment to add inputs:
        # "fertilizer_schedule": [
        #     {"days_after_planting": 15, "n_kg_ha": 60.0, "p_kg_ha": 30.0},
        #     {"days_after_planting": 45, "n_kg_ha": 30.0, "p_kg_ha":  0.0},
        # ]
    },
}

with open(str(SIM_CONFIG), "w") as fh:
    yaml.dump(sim_config, fh, default_flow_style=False, sort_keys=False)
print(f"  Config: {SIM_CONFIG}")

from ag_cube_cm.mcp_server import run_simulation

result_json = run_simulation(
    config_path=str(SIM_CONFIG),
    max_pixels=None,   # pass e.g. max_pixels=20 for a quick sanity-check
    feature=None,      # e.g. "Comayagua" to run on one district only
    adm_level=2,
)
result = json.loads(result_json)

if result["status"] != "ok":
    print(f"\n[ERROR] Simulation failed:\n{result['message']}")
    sys.exit(1)

print(f"\n  Done.")
print(f"  Pixels OK       : {result['pixels_ok']}")
print(f"  Pixels skipped  : {result['pixels_skipped']}")
print(f"  Pixels failed   : {result['pixels_failed']}")
print(f"  Planting windows: {result['n_planting_windows']}")
print(f"  Years simulated : {result['n_years']}  {result['years']}")
print(f"  Mean HWAM       : {result['mean_hwam_kg_ha']} kg/ha")
print(f"  Output          : {result['output_path']}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Visualize: potential yield and inter-annual variability maps
# ──────────────────────────────────────────────────────────────────────────────

print("\n[7/7] Plotting results …")

import matplotlib.pyplot as plt
import numpy as np

yield_ds = xr.open_dataset(result["output_path"])
hwam = yield_ds["HWAM"]   # (planting_window, year, y, x)   kg/ha

# Mean yield across all planting windows and all simulated years
mean_yield = hwam.mean(dim=["planting_window", "year"])

# Coefficient of variation (%): inter-annual variability averaged across windows
std_yield  = hwam.std(dim="year").mean(dim="planting_window")
cv_yield   = (std_yield / mean_yield * 100).where(mean_yield > 0)

# Best planting window: window index that maximises mean yield
best_window = hwam.mean(dim="year").argmax(dim="planting_window")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"{CROP} potential yield — Honduras Comayagua  "
    f"({START_DATE[:4]}–{END_DATE[:4]}, {N_WINDOWS} planting windows, DSSAT {CULTIVAR})",
    fontsize=11,
)

# Panel 1 — mean potential yield
mean_yield.plot(
    ax=axes[0], cmap="YlGn", vmin=0,
    cbar_kwargs={"label": "Mean yield  (kg/ha)"},
)
axes[0].set_title("Mean potential yield")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

# Panel 2 — inter-annual yield variability (CV %)
cv_yield.plot(
    ax=axes[1], cmap="RdYlGn_r", vmin=0, vmax=60,
    cbar_kwargs={"label": "CV yield  (%)"},
)
axes[1].set_title("Inter-annual variability (CV %)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("")

# Panel 3 — optimal planting window index (0-based)
best_window.plot(
    ax=axes[2], cmap="viridis",
    cbar_kwargs={"label": f"Best window (0 = {PLANTING_DATE[5:]})"},
)
axes[2].set_title(f"Optimal planting window\n(0 = {PLANTING_DATE[5:]}, step = {WINDOW_DAYS} d)")
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("")

plt.tight_layout()
out_png = BASE_DIR / "yield_map.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.show()
print(f"  → {out_png}")

yield_ds.close()
print("\nDone.  Output files:")
print(f"  Weather datacube : {WEATHER_NC}")
print(f"  Soil datacube    : {SOIL_NC}")
print(f"  Yield output     : {YIELD_NC}")
print(f"  Map              : {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE CLIMATE SOURCE — NASA POWER (no API key, 0.5° resolution)
# ──────────────────────────────────────────────────────────────────────────────
# Useful for a quick test or when no CDS API key is available.
# Replace STEPS 1–3 with:
#
#   from aggeodata.ingestion.nasa_power import NASAPowerDownloader
#
#   nasa_params = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN"]
#   dl = NASAPowerDownloader(parameters=nasa_params)
#   nasa_nc = dl.download(
#       extent=BBOX,
#       starting_date=START_DATE,
#       ending_date=END_DATE,
#       output_folder=str(CLIMATE_DIR),
#   )
#   # NASA POWER delivers native codes — rename to DSSAT names before saving:
#   raw = xr.open_dataset(nasa_nc)
#   weather_ds = raw.rename({
#       "PRECTOTCORR":       "precipitation",
#       "T2M_MAX":           "tmax",
#       "T2M_MIN":           "tmin",
#       "ALLSKY_SFC_SW_DWN": "solar_radiation",
#   })
#   time_dim = next(d for d in weather_ds.dims if d in {"time","date"})
#   if time_dim != "date":
#       weather_ds = weather_ds.rename({time_dim: "date"})
#   weather_ds.rio.write_crs("EPSG:4326").to_netcdf(str(WEATHER_NC))
#
#   # Then continue from STEP 4 (soil) as normal.
#   # Note: with NASA POWER names already mapped to DSSAT names, the CF-alias
#   # step in DSSATModel.prepare_inputs() is a no-op — both paths are safe.
