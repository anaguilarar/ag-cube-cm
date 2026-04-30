"""
ag-cube-cm MCP server
=====================
Exposes the package's core operations as MCP tools so an AI assistant
(Claude + spatial-crop-modeler skill) can orchestrate full workflows via
natural language:

  download_weather  → download_soil  → generate_config  → run_simulation

Start the server:
    python -m ag_cube_cm.mcp_server

Or register it in .claude/mcp_config.json (see project README).
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ag-cube-cm",
    instructions=(
        "Tools for downloading AgERA5/CHIRPS weather, SoilGrids soil data, "
        "and running DSSAT/BANANA_N crop model simulations on spatial datacubes."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _country_bbox(country_code: str, adm_level: int = 0) -> list[float]:
    """Return [xmin, ymin, xmax, ymax] for a country via GeoBoundaries API."""
    import requests
    url = f"https://www.geoboundaries.org/api/current/gbOpen/{country_code.upper()}/ADM{adm_level}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    geojson_url = data.get("gjDownloadURL", "")
    gj = requests.get(geojson_url, timeout=60).json()
    import geopandas as gpd, io, json as _json
    gdf = gpd.read_file(io.StringIO(_json.dumps(gj)))
    b = gdf.total_bounds  # [xmin, ymin, xmax, ymax]
    return [round(float(b[0]), 4), round(float(b[1]), 4),
            round(float(b[2]), 4), round(float(b[3]), 4)]


def _ok(payload: Any) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg})


# ---------------------------------------------------------------------------
# Tool 1 — download_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def download_weather(
    country_code: str,
    year_start: int,
    year_end: int,
    source: str = "agera5",
    output_folder: str | None = None,
    bbox: list[float] | None = None,
) -> str:
    """Download weather data for a country and merge into a single NetCDF datacube.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. 'MWI', 'HND', 'GLP').
    year_start / year_end : int
        Inclusive year range (e.g. 2000, 2019).
    source : str
        'agera5' (temperature, solar radiation, wind, humidity) or
        'chirps' (rainfall only). Default 'agera5'.
    output_folder : str | None
        Where to save downloaded files and the final NetCDF.
        Defaults to '<tempdir>/ag_cube_cm/<country_code>/weather'.
    bbox : list[float] | None
        [xmin, ymin, xmax, ymax] override. Auto-fetched from GeoBoundaries
        when omitted.

    Returns
    -------
    JSON with status, output_path, and dataset dimensions.
    """
    try:
        from ag_cube_cm.ingestion.weather import WeatherDownloadOrchestrator

        if output_folder is None:
            output_folder = str(
                Path(tempfile.gettempdir()) / "ag_cube_cm" / country_code.upper() / "weather"
            )
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        if bbox is None:
            bbox = _country_bbox(country_code)

        starting_date = f"{year_start}-01-01"
        ending_date   = f"{year_end}-12-31"

        orch = WeatherDownloadOrchestrator(
            starting_date=starting_date,
            ending_date=ending_date,
            xyxy=bbox,
            output_folder=output_folder,
        )

        if source.lower() == "chirps":
            variables = {
                "precipitation": {"mission": "chirps", "source": "chirps"},
            }
        else:
            variables = {
                "temperature_2m_max":       {"mission": "agera5", "source": "agera5"},
                "temperature_2m_min":       {"mission": "agera5", "source": "agera5"},
                "solar_radiation_flux":     {"mission": "agera5", "source": "agera5"},
                "precipitation_flux":       {"mission": "agera5", "source": "agera5"},
                "wind_speed":               {"mission": "agera5", "source": "agera5"},
            }

        results = orch.download(variables, export_as_netcdf=True)

        nc_files = [v for vdict in results.values() for v in vdict.values()
                    if str(v).endswith(".nc")]
        output_path = nc_files[0] if nc_files else output_folder

        return _ok({
            "output_path": output_path,
            "country_code": country_code,
            "bbox": bbox,
            "year_range": [year_start, year_end],
            "source": source,
            "files_downloaded": len(nc_files),
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")


# ---------------------------------------------------------------------------
# Tool 2 — download_soil
# ---------------------------------------------------------------------------

@mcp.tool()
def download_soil(
    country_code: str,
    output_folder: str | None = None,
    bbox: list[float] | None = None,
    depths: list[str] | None = None,
    variables: list[str] | None = None,
) -> str:
    """Download SoilGrids data and merge into a multi-depth NetCDF datacube.

    Downloads raw GeoTIFF files from SoilGrids, then builds and saves a
    merged multi-depth NetCDF datacube ready for use with run_simulation.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. 'MWI').
    output_folder : str | None
        Where to save downloaded files and the final NetCDF.
        Defaults to '<tempdir>/ag_cube_cm/<country_code>/soil'.
    bbox : list[float] | None
        [xmin, ymin, xmax, ymax] override.  Auto-fetched when omitted.
    depths : list[str] | None
        Depth intervals. Defaults to ["0-5", "5-15", "15-30", "30-60", "60-100"].
    variables : list[str] | None
        SoilGrids variables. Defaults to the standard DSSAT set:
        clay, sand, silt, bdod, cfvo, nitrogen, phh2o, soc, wv0010, wv0033, wv1500.

    Returns
    -------
    JSON with status, output_path (the merged NetCDF), output_folder, and file count.
    """
    try:
        from ag_cube_cm.ingestion.soil import SoilGridsDownloader
        from ag_cube_cm.transform.soil_cube import SoilDataCubeBuilder

        if output_folder is None:
            output_folder = str(
                Path(tempfile.gettempdir()) / "ag_cube_cm" / country_code.upper() / "soil"
            )
        if depths is None:
            depths = ["0-5", "5-15", "15-30", "30-60", "60-100"]
        if variables is None:
            variables = ["clay", "sand", "silt", "bdod", "cfvo",
                         "nitrogen", "phh2o", "soc", "wv0010", "wv0033", "wv1500"]
        if bbox is None:
            bbox = _country_bbox(country_code)

        # Step 1 — download raw GeoTIFFs from SoilGrids
        dl = SoilGridsDownloader(
            soil_layers=variables,
            depths=depths,
            output_folder=output_folder,
        )
        downloaded = dl.download(boundaries=bbox)

        # Step 2 — merge GeoTIFFs into a multi-depth NetCDF datacube
        nc_filename = f"soil_{country_code.lower()}.nc"
        builder = SoilDataCubeBuilder(
            data_folder=output_folder,
            variables=variables,
            extent=bbox,
            reference_variable="wv1500",
            target_crs="EPSG:4326",
        )
        nc_path = builder.build_and_save(
            output_path=output_folder,
            filename=nc_filename,
        )

        return _ok({
            "output_path": nc_path,
            "output_folder": output_folder,
            "country_code": country_code,
            "bbox": bbox,
            "depths": depths,
            "variables": variables,
            "files_downloaded": len(downloaded),
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")


# ---------------------------------------------------------------------------
# Tool 3 — generate_config
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_config(
    country: str,
    country_code: str,
    model: str,
    weather_path: str,
    soil_path: str,
    crop: str,
    cultivar: str,
    planting_date: str,
    output_path: str,
    working_path: str,
    dssat_path: str | None = None,
    n_planting_windows: int = 1,
    planting_window_days: int = 7,
    ncores: int = 4,
    fertilizer_n_kg_ha: float = 0.0,
    fertilizer_p_kg_ha: float = 0.0,
    feature: str | None = None,
    adm_level: int = 2,
    save_to: str | None = None,
) -> str:
    """Generate and optionally save a simulation YAML config file.

    Parameters
    ----------
    country : str           Full country name (e.g. 'Malawi').
    country_code : str      ISO 3-letter code (e.g. 'MWI').
    model : str             'dssat', 'banana_n', 'simple_model', or 'caf'.
    weather_path : str      Path to the weather NetCDF datacube.
    soil_path : str         Path to the soil NetCDF datacube.
    crop : str              Crop name (e.g. 'Maize', 'Wheat', 'Bean').
    cultivar : str          DSSAT cultivar ID (e.g. 'IB1072').
    planting_date : str     Base planting date 'YYYY-MM-DD'.
    output_path : str       Where to save the yield NetCDF output.
    working_path : str      DSSAT run working directory (NO spaces in path).
    dssat_path : str | None DSSAT installation root (None = use bundled binary).
    n_planting_windows : int  Number of planting windows to simulate.
    planting_window_days : int  Days between consecutive windows.
    ncores : int            Parallel threads.
    fertilizer_n_kg_ha : float  N applied at planting (kg/ha). 0 = no fertilizer.
    fertilizer_p_kg_ha : float  P applied at planting (kg/ha).
    feature : str | None    Admin unit to restrict the simulation to
                            (e.g. 'Zomba', 'Comayagua').  None = full country.
    adm_level : int         Admin level for the feature boundary (default 2).
    save_to : str | None    If given, writes the YAML to this file path.

    Returns
    -------
    JSON with status, config_yaml (string), and save_path.
    """
    try:
        fert_block = ""
        if fertilizer_n_kg_ha > 0 or fertilizer_p_kg_ha > 0:
            fert_block = (
                "  fertilizer_schedule:\n"
                f"    - days_after_planting: 5\n"
                f"      n_kg_ha: {fertilizer_n_kg_ha}\n"
                f"      p_kg_ha: {fertilizer_p_kg_ha}\n"
            )

        dssat_line = (
            f"  dssat_path: '{dssat_path}'\n" if dssat_path
            else "  dssat_path: null\n"
        )

        feature_line = (
            f"  feature: '{feature}'\n" if feature else "  feature: null\n"
        )

        yaml_text = (
            f"GENERAL_INFO:\n"
            f"  country: '{country}'\n"
            f"  country_code: '{country_code.upper()}'\n"
            f"  model: '{model}'\n"
            f"  working_path: '{working_path}'\n"
            f"{dssat_line}"
            f"  ncores: {ncores}\n"
            f"\n"
            f"SPATIAL_INFO:\n"
            f"  feature_name: 'shapeName'\n"
            f"  adm_level: {adm_level}\n"
            f"{feature_line}"
            f"  soil_path: '{soil_path}'\n"
            f"  weather_path: '{weather_path}'\n"
            f"  output_path: '{output_path}'\n"
            f"  dem_path: null\n"
            f"\n"
            f"CROP:\n"
            f"  name: '{crop}'\n"
            f"  cultivar: '{cultivar}'\n"
            f"\n"
            f"MANAGEMENT:\n"
            f"  planting_date: '{planting_date}'\n"
            f"  n_planting_windows: {n_planting_windows}\n"
            f"  planting_window_days: {planting_window_days}\n"
            f"{fert_block}"
        )

        save_path = None
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, "w") as fh:
                fh.write(yaml_text)
            save_path = save_to

        return _ok({"config_yaml": yaml_text, "save_path": save_path})
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tool 4 — list_admin_units
# ---------------------------------------------------------------------------

@mcp.tool()
def list_admin_units(
    country_code: str,
    adm_level: int = 2,
) -> str:
    """List all administrative unit names for a country at a given level.

    Useful for discovering valid values for the *feature* parameter of
    run_simulation and generate_config before running a simulation.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. 'MWI', 'HND', 'COL').
    adm_level : int
        Administrative level.  1 = region/province, 2 = district/department
        (default), 3 = sub-district.

    Returns
    -------
    JSON with status, country_code, adm_level, count, and sorted list of names.
    """
    try:
        from ag_cube_cm.ingestion.boundaries import list_admin_units as _list

        names = _list(country_code, adm_level=adm_level)
        return _ok({
            "country_code": country_code.upper(),
            "adm_level": adm_level,
            "count": len(names),
            "units": names,
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")


# ---------------------------------------------------------------------------
# Tool 5 — run_simulation
# ---------------------------------------------------------------------------

@mcp.tool()
def run_simulation(
    config_path: str,
    max_pixels: int | None = None,
    feature: str | None = None,
    adm_level: int = 2,
) -> str:
    """Run a spatial crop model simulation from a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to the simulation YAML config (generated by generate_config or
        written manually).
    max_pixels : int | None
        Optional cap on number of pixels to simulate (useful for quick tests).
        None = run all land pixels.
    feature : str | None
        Admin unit name to restrict the simulation to (e.g. 'Zomba', 'Comayagua').
        Overrides the feature set in the config file.  None = use the entire
        country bounding box from the config.
    adm_level : int
        Administrative level for the feature boundary lookup.
        1 = region/province, 2 = district/department (default), 3 = sub-district.

    Returns
    -------
    JSON with status, output_path, pixel counts (ok/skip/failed), and
    mean HWAM yield when available.
    """
    try:
        import concurrent.futures
        from datetime import timedelta

        import numpy as np
        import pandas as pd
        import xarray as xr

        from ag_cube_cm.config.loader import load_config
        from ag_cube_cm.models.dssat.base import DSSATModel

        cfg = load_config(config_path)
        weather_ds = xr.open_dataset(cfg.SPATIAL_INFO.weather_path)
        soil_ds    = xr.open_dataset(cfg.SPATIAL_INFO.soil_path)

        # Clip to admin boundary if requested (via parameter or config)
        effective_feature = feature or cfg.SPATIAL_INFO.feature
        if effective_feature:
            from ag_cube_cm.ingestion.boundaries import get_admin_boundary
            from ag_cube_cm.spatial.raster_ops import get_roi_data

            effective_adm = adm_level if feature else cfg.SPATIAL_INFO.adm_level
            boundary_gdf = get_admin_boundary(
                cfg.GENERAL_INFO.country_code, effective_feature,
                adm_level=effective_adm,
            )
            weather_ds = get_roi_data(weather_ds, boundary_gdf)
            soil_ds    = get_roi_data(soil_ds,    boundary_gdf)

        base_pdate = cfg.MANAGEMENT.planting_date
        n_windows  = cfg.MANAGEMENT.n_planting_windows or 1
        step       = cfg.MANAGEMENT.planting_window_days
        planting_dates = [base_pdate + timedelta(days=w * step) for w in range(n_windows)]

        # Build pixel list
        pixel_coords: dict[int, tuple[float, float]] = {
            idx: (float(y), float(x))
            for idx, (y, x) in enumerate(
                (y, x)
                for y in weather_ds.y.values
                for x in weather_ds.x.values
            )
        }

        if max_pixels is not None:
            pixel_coords = dict(list(pixel_coords.items())[:max_pixels])

        ncores = cfg.GENERAL_INFO.ncores

        def _run(args):
            pixel_idx, w_idx, pdate, y, x = args
            dir_name = f"px{pixel_idx}_w{w_idx:02d}"
            res = {"pixel_idx": pixel_idx, "window_idx": w_idx,
                   "y": y, "x": x, "HWAM": np.nan, "flag": 2, "error": ""}
            try:
                wsl = weather_ds.sel(y=y, x=x, method="nearest")
                ssl = soil_ds.sel(y=y, x=x, method="nearest")
                if (wsl.to_dataframe().reset_index().dropna().empty or
                        ssl.to_dataframe().reset_index().dropna().empty):
                    res["error"] = "no-data pixel"
                    return res
                mgmt_w = cfg.MANAGEMENT.model_copy(update={"planting_date": pdate})
                cfg_w  = cfg.model_copy(update={"MANAGEMENT": mgmt_w})
                model  = DSSATModel(cfg_w)
                model.setup_working_directory(dir_name)
                try:
                    model.prepare_inputs(wsl, ssl, elevation=0.0)
                    model.run_simulation()
                    outputs = model.collect_outputs()
                    hwam = outputs.get("HWAM", np.nan)
                    res["HWAM"] = float(hwam) if hwam not in (None, "", "-99") else np.nan
                    res["flag"] = 0 if outputs else 1
                    if not outputs:
                        res["error"] = "DSSAT produced no output"
                except Exception as e:
                    res["flag"] = 1
                    res["error"] = str(e)
                finally:
                    model.cleanup_working_directory()
            except Exception as e:
                res["flag"] = 1
                res["error"] = str(e)
            return res

        jobs = [
            (idx, w_idx, pdate, pixel_coords[idx][0], pixel_coords[idx][1])
            for idx in pixel_coords
            for w_idx, pdate in enumerate(planting_dates)
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
            for r in pool.map(_run, jobs):
                results.append(r)

        weather_ds.close()
        soil_ds.close()

        df = pd.DataFrame(results)
        ok_df = df[df["flag"] == 0]
        n_ok     = len(ok_df)
        n_skip   = int((df["flag"] == 2).sum())
        n_failed = int((df["flag"] == 1).sum())
        mean_hwam = float(ok_df["HWAM"].dropna().mean()) if n_ok else None

        # Build output NetCDF
        y_vals = sorted(df["y"].unique())
        x_vals = sorted(df["x"].unique())
        yi_map = {v: i for i, v in enumerate(y_vals)}
        xi_map = {v: i for i, v in enumerate(x_vals)}
        hwam_grid = np.full((n_windows, len(y_vals), len(x_vals)), np.nan, dtype=np.float32)
        flag_grid = np.full((n_windows, len(y_vals), len(x_vals)), 2, dtype=np.int8)
        for _, row in df.iterrows():
            wi = int(row["window_idx"])
            yi = yi_map[row["y"]]
            xi = xi_map[row["x"]]
            hwam_grid[wi, yi, xi] = row["HWAM"]
            flag_grid[wi, yi, xi] = int(row["flag"])

        ds_out = xr.Dataset(
            {
                "HWAM": (["planting_window", "y", "x"], hwam_grid,
                         {"long_name": "Mean grain yield at maturity", "units": "kg/ha"}),
                "flag": (["planting_window", "y", "x"], flag_grid,
                         {"long_name": "0=ok 1=failed 2=no_data"}),
            },
            coords={
                "planting_window": np.arange(n_windows),
                "planting_date": (["planting_window"],
                                  [str(p) for p in planting_dates]),
                "y": y_vals,
                "x": x_vals,
            },
        )
        out = cfg.SPATIAL_INFO.output_path
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            import os as _os
            if _os.path.exists(out):
                _os.remove(out)
            ds_out.to_netcdf(out)

        return _ok({
            "output_path": out,
            "pixels_ok": n_ok,
            "pixels_skipped": n_skip,
            "pixels_failed": n_failed,
            "n_planting_windows": n_windows,
            "mean_hwam_kg_ha": round(mean_hwam, 1) if mean_hwam else None,
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")


# ---------------------------------------------------------------------------
# Tool 6 — list_supported_crops  (informational)
# ---------------------------------------------------------------------------

@mcp.tool()
def list_supported_crops() -> str:
    """List all crops supported by the DSSAT model and their 2-letter codes."""
    crops = {
        "Maize": "MZ", "Wheat": "WH", "Rice": "RI", "Sorghum": "SG",
        "Millet": "ML", "Soybean": "SB", "Bean": "BN", "Cassava": "CS",
        "Potato": "PT", "Sugarcane": "SC", "Sugarbeet": "BS",
        "Sunflower": "SU", "Canola": "CN", "Tomato": "TM",
        "Cabbage": "CB", "Alfalfa": "AL", "Bermudagrass": "BM",
    }
    common_cultivars = {
        "Maize":  ["IB1072 (tropical)", "PC0002 (temperate)", "MEDIUM (generic)"],
        "Wheat":  ["IB1015 (spring)", "IB1487 (winter)"],
        "Bean":   ["IB0001"],
        "Soybean":["IB0001"],
    }
    return _ok({"crops": crops, "example_cultivars": common_cultivars})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
