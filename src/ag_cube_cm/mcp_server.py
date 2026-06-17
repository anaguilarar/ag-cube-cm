"""
ag-cube-cm MCP server
=====================
Exposes spatial crop-model operations as MCP tools.

Typical workflow (use the **aggeodata** MCP server first for data downloads):
  aggeodata: download_chirps / download_agera5 / download_soil
  aggeodata: build_climate_datacube / build_soil_datacube
  ag-cube-cm: generate_config  →  run_simulation

Start the server:
    python -m ag_cube_cm.mcp_server
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ag-cube-cm",
    instructions=(
        "Tools for running DSSAT/BANANA_N crop model simulations on spatial "
        "datacubes produced by the aggeodata package. Use the aggeodata MCP "
        "server first to download climate and soil data."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(payload: Any) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg})


# ---------------------------------------------------------------------------
# Tool 1 — list_admin_units
# ---------------------------------------------------------------------------

@mcp.tool()
def list_admin_units(
    country_code: str,
    adm_level: int = 2,
) -> str:
    """List all administrative unit names for a country at a given level.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. 'MWI', 'HND', 'COL').
    adm_level : int
        1 = region/province, 2 = district/department (default), 3 = sub-district.

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
# Tool 2 — generate_config
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
    weather_path : str      Path to the climate NetCDF datacube from aggeodata.
    soil_path : str         Path to the soil NetCDF datacube from aggeodata.
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
    feature : str | None    Admin unit to restrict the simulation to.
    adm_level : int         Admin level for the feature boundary (default 2).
    save_to : str | None    If given, writes the YAML to this file path.

    Returns
    -------
    JSON with status, config_yaml (string), and save_path.
    """
    try:
        space_warning = (
            f"WARNING: working_path '{working_path}' contains spaces. "
            "DSSAT will silently fail (rc=99). Use a path without spaces."
            if " " in working_path else None
        )

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

        payload: dict[str, Any] = {"config_yaml": yaml_text, "save_path": save_path}
        if space_warning:
            payload["warning"] = space_warning
        return _ok(payload)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tool 3 — list_supported_crops
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
        "Maize":   ["IB1072 (tropical)", "PC0002 (temperate)", "MEDIUM (generic)"],
        "Wheat":   ["IB1015 (spring)", "IB1487 (winter)"],
        "Bean":    ["IB0001"],
        "Soybean": ["IB0001"],
    }
    return _ok({"crops": crops, "example_cultivars": common_cultivars})


# ---------------------------------------------------------------------------
# Tool 4 — run_simulation
# ---------------------------------------------------------------------------

@mcp.tool()
def run_simulation(
    config_path: str,
    max_pixels: int | None = None,
    feature: str | None = None,
    adm_level: int = 2,
) -> str:
    """Run a spatial crop model simulation from a YAML config file.

    Prerequisites
    -------------
    Climate and soil NetCDF datacubes must already exist on disk (produced by
    the **aggeodata** MCP server tools ``build_climate_datacube`` and
    ``build_soil_datacube``).

    Parameters
    ----------
    config_path : str
        Path to the simulation YAML config (generated by generate_config or
        written manually).
    max_pixels : int | None
        Optional cap on number of pixels to simulate (useful for quick tests).
        None = run all land pixels.
    feature : str | None
        Admin unit name to restrict the simulation to (e.g. 'Zomba').
        Overrides the feature set in the config file.
    adm_level : int
        Administrative level for the feature boundary lookup (default 2).

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

        import rioxarray as _rio  # noqa: F401
        for _ds in (weather_ds, soil_ds):
            for _vname in list(_ds.data_vars) + list(_ds.coords):
                _ds.variables[_vname].encoding.pop("grid_mapping", None)
        weather_ds = weather_ds.rio.write_crs("EPSG:4326", inplace=True)
        soil_ds    = soil_ds.rio.write_crs("EPSG:4326", inplace=True)

        def _normalise_dims(ds: xr.Dataset) -> xr.Dataset:
            rn: dict[str, str] = {}
            if "lat" in ds.dims and "y" not in ds.dims:
                rn["lat"] = "y"
            if "lon" in ds.dims and "x" not in ds.dims:
                rn["lon"] = "x"
            if "latitude" in ds.dims and "y" not in ds.dims:
                rn["latitude"] = "y"
            if "longitude" in ds.dims and "x" not in ds.dims:
                rn["longitude"] = "x"
            return ds.rename(rn) if rn else ds

        weather_ds = _normalise_dims(weather_ds)
        soil_ds    = _normalise_dims(soil_ds)

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

        time_dim = next(
            (d for d in weather_ds.dims if d in {"time", "date"}), None
        )
        if time_dim is not None:
            try:
                raw_times = weather_ds[time_dim].values
                all_years = sorted({int(pd.Timestamp(t).year) for t in raw_times})
            except Exception:
                all_years = [base_pdate.year]
        else:
            all_years = [base_pdate.year]

        last_yr = max(all_years)
        sim_years = []
        for yr in all_years:
            try:
                pdate_yr = base_pdate.replace(year=yr)
            except ValueError:
                pdate_yr = base_pdate.replace(year=yr, day=28)
            if (pdate_yr + timedelta(days=200)).year <= last_yr:
                sim_years.append(yr)
        if not sim_years:
            sim_years = all_years

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

        def _run(args: tuple) -> dict:
            pixel_idx, w_idx, yr, pdate_yr, y, x = args
            dir_name = f"px{pixel_idx}_w{w_idx:02d}_y{yr}"
            res: dict = {
                "pixel_idx": pixel_idx, "window_idx": w_idx, "year": yr,
                "y": y, "x": x, "HWAM": np.nan, "flag": 2, "error": "",
            }
            try:
                wsl = weather_ds.sel(y=y, x=x, method="nearest")
                ssl = soil_ds.sel(y=y, x=x, method="nearest")

                if time_dim is not None:
                    try:
                        times = pd.DatetimeIndex(
                            [pd.Timestamp(t) for t in wsl[time_dim].values]
                        )
                        harvest_yr = (pdate_yr + timedelta(days=250)).year
                        yr_mask = times.year.isin(range(yr, harvest_yr + 1))
                        wsl = wsl.isel({time_dim: yr_mask.values})
                    except Exception:
                        pass

                if (wsl.to_dataframe().reset_index().dropna().empty or
                        ssl.to_dataframe().reset_index().dropna().empty):
                    res["error"] = "no-data pixel"
                    return res

                mgmt_w = cfg.MANAGEMENT.model_copy(update={"planting_date": pdate_yr})
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

        jobs = []
        for idx in pixel_coords:
            py, px = pixel_coords[idx]
            for w_idx, base_pd in enumerate(planting_dates):
                for yr in sim_years:
                    try:
                        pdate_yr = base_pd.replace(year=yr)
                    except ValueError:
                        pdate_yr = base_pd.replace(year=yr, day=28)
                    jobs.append((idx, w_idx, yr, pdate_yr, py, px))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
            for r in pool.map(_run, jobs):
                results.append(r)

        weather_ds.close()
        soil_ds.close()

        df = pd.DataFrame(results)
        ok_df    = df[df["flag"] == 0]
        n_ok     = len(ok_df)
        n_skip   = int((df["flag"] == 2).sum())
        n_failed = int((df["flag"] == 1).sum())
        mean_hwam = float(ok_df["HWAM"].dropna().mean()) if n_ok else None

        y_vals   = sorted(df["y"].unique())
        x_vals   = sorted(df["x"].unique())
        yi_map   = {v: i for i, v in enumerate(y_vals)}
        xi_map   = {v: i for i, v in enumerate(x_vals)}
        yr_i_map = {yr: i for i, yr in enumerate(sim_years)}

        hwam_grid = np.full(
            (n_windows, len(sim_years), len(y_vals), len(x_vals)),
            np.nan, dtype=np.float32,
        )
        flag_grid = np.full(
            (n_windows, len(sim_years), len(y_vals), len(x_vals)),
            2, dtype=np.int8,
        )
        for _, row in df.iterrows():
            wi  = int(row["window_idx"])
            yri = yr_i_map[int(row["year"])]
            yi  = yi_map[row["y"]]
            xi  = xi_map[row["x"]]
            hwam_grid[wi, yri, yi, xi] = row["HWAM"]
            flag_grid[wi, yri, yi, xi] = int(row["flag"])

        ds_out = xr.Dataset(
            {
                "HWAM": (["planting_window", "year", "y", "x"], hwam_grid,
                         {"long_name": "Mean grain yield at maturity", "units": "kg/ha"}),
                "flag": (["planting_window", "year", "y", "x"], flag_grid,
                         {"long_name": "0=ok 1=failed 2=no_data"}),
            },
            coords={
                "planting_window": np.arange(n_windows),
                "year": sim_years,
                "planting_date": (["planting_window"],
                                  [str(p) for p in planting_dates]),
                "y": y_vals,
                "x": x_vals,
            },
        )

        for _vname in list(ds_out.data_vars) + list(ds_out.coords):
            ds_out.variables[_vname].encoding.pop("grid_mapping", None)
        ds_out = ds_out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds_out = ds_out.rio.write_crs("EPSG:4326")
        ds_out["x"].attrs.update({
            "standard_name": "longitude", "long_name": "longitude",
            "units": "degrees_east", "axis": "X",
        })
        ds_out["y"].attrs.update({
            "standard_name": "latitude", "long_name": "latitude",
            "units": "degrees_north", "axis": "Y",
        })
        ds_out.attrs["Conventions"] = "CF-1.8"

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
            "n_years": len(sim_years),
            "years": sim_years,
            "mean_hwam_kg_ha": round(mean_hwam, 1) if mean_hwam is not None else None,
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
