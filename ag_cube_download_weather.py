#!/usr/bin/env python
"""
ag_cube_download_weather.py
============================

Drop-in replacement for the legacy ``download_weather_data.py``.

Usage (identical CLI to the legacy script)::

    # Download raw yearly files:
    python ag_cube_download_weather.py --config options/ag_cube_weather_config.yaml

    # Build the multi-temporal datacube (set task: datacube in the YAML):
    python ag_cube_download_weather.py --config options/ag_cube_weather_config.yaml

What changed vs the legacy script
----------------------------------
| Legacy                          | New (ag-cube-cm)                          |
|---------------------------------|-------------------------------------------|
| ``ClimateDataDownload``         | ``WeatherDownloadOrchestrator``           |
| ``MLTWeatherDataCube``          | ``MLTWeatherDataCube`` (same API)         |
| ``stack_datacube_temporally``   | ``stack_datacube_temporally`` (same)      |
| ``set_encoding``                | ``set_weather_encoding``                  |
| Raw OmegaConf access            | OmegaConf still used for loading;         |
|                                 | Pydantic validation added on top          |
| Flat ``import spatialdata.*``   | ``ag_cube_cm.ingestion.weather`` +        |
|                                 | ``ag_cube_cm.transform.weather_cube``     |

The YAML format is **unchanged** — your existing config files work as-is.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# ag-cube-cm imports  (replaces the legacy spatialdata.* imports)
# ---------------------------------------------------------------------------
from ag_cube_cm.ingestion.weather import WeatherDownloadOrchestrator
from ag_cube_cm.transform.weather_cube import (
    MLTWeatherDataCube,
    stack_datacube_temporally,
    set_weather_encoding,
    METEO_NAMES,
)
from ag_cube_cm.spatial.raster_ops import get_boundaries_from_shapefile
from ag_cube_cm.ingestion.files_manager import IntervalFolderManager   # still needed internally

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and/or build AgEra5 / CHIRPS weather datacubes "
                    "using the ag-cube-cm package."
    )
    parser.add_argument(
        "--config",
        default="options/ag_cube_weather_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("ag-cube-cm weather pipeline starting …")
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load configuration (OmegaConf keeps ${} interpolation support)
    # -----------------------------------------------------------------------
    config = OmegaConf.load(args.config)
    logger.info("Configuration loaded from '%s'.", args.config)

    # -----------------------------------------------------------------------
    # Resolve spatial extent
    # -----------------------------------------------------------------------
    spatial_file = config.SPATIAL_INFO.get("spatial_file", None)
    if spatial_file:
        # Use raster_ops helper instead of the legacy get_boundaries_from_path
        extent = list(
            get_boundaries_from_shapefile(spatial_file, round_numbers=True)
        )
        logger.info("Extent derived from shapefile: %s", extent)
    else:
        extent = list(config.SPATIAL_INFO.extent)
        logger.info("Extent from config: %s", extent)

    output_path: str = config.PATHS.output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # TASK: download
    # Download raw yearly AgEra5 zip files and/or CHIRPS daily NetCDF files.
    # -----------------------------------------------------------------------
    if config.GENERAL.task == "download":
        logger.info("Task: DOWNLOAD")

        orchestrator = WeatherDownloadOrchestrator(
            starting_date=config.DATES.starting_date,
            ending_date=config.DATES.ending_date,
            xyxy=extent,
            output_folder=output_path,
        )

        # weather_variables is a DictConfig — convert to plain dict for the call
        weather_vars = OmegaConf.to_container(config.WEATHER.variables, resolve=True)

        orchestrator.download(
            weather_variables=weather_vars,
            suffix=config.GENERAL.suffix,
            export_as_netcdf=config.GENERAL.get("export_as_netcdf", False),
            ncores=config.GENERAL.get("ncores", 4),
            version=config.GENERAL.get("agera5_version", "2_0"),
        )

        logger.info("Download completed!")

    # -----------------------------------------------------------------------
    # TASK: datacube
    # Stack the downloaded per-date files into one multi-temporal NetCDF cube.
    # -----------------------------------------------------------------------
    elif config.GENERAL.task == "datacube":
        logger.info("Task: DATACUBE")

        # Build the {variable: folder_path} map using the same naming convention
        # as the legacy script: <var>_<suffix>_raw
        suffix = config.GENERAL.suffix
        list_weather_paths: dict[str, str] = {}
        for raw_key, short_name in METEO_NAMES.items():
            if raw_key in OmegaConf.to_container(config.WEATHER.variables):
                folder = os.path.join(output_path, f"{raw_key}_{suffix}_raw")
                list_weather_paths[short_name] = folder

        logger.info("Variable folders: %s", list_weather_paths)

        # Build the multi-temporal cube using MLTWeatherDataCube
        folder_manager = IntervalFolderManager()
        cube_builder = MLTWeatherDataCube(
            directory_paths=list_weather_paths,
            folder_manager=folder_manager,
        )

        cube_builder.common_dates_and_file_names(
            starting_date=config.DATES.starting_date,
            ending_date=config.DATES.ending_date,
        )

        mlt_data = cube_builder.multitemporal_data(
            reference_variable=config.GENERAL.reference_variable
        )

        # Stack the per-date dict into a single Dataset along the 'date' axis
        weather_cube = stack_datacube_temporally(
            mlt_data, time_dim_name="date", parse_dates=True
        )

        # Build output file name:  weather_<suffix>_<start_year>_<end_year>.nc
        sy = config.DATES.starting_date[:4]
        ey = config.DATES.ending_date[:4]
        out_nc = os.path.join(output_path, f"weather_{suffix}_{sy}_{ey}.nc")

        encoding = set_weather_encoding(weather_cube)
        weather_cube.to_netcdf(out_nc, encoding=encoding, engine="netcdf4")
        logger.info("Weather datacube saved → %s", out_nc)

    else:
        raise ValueError(
            f"Unknown task '{config.GENERAL.task}'. "
            "Expected 'download' or 'datacube'."
        )


if __name__ == "__main__":
    main()
