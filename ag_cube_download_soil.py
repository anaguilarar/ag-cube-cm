#!/usr/bin/env python
"""
ag_cube_download_soil.py
=========================

Drop-in replacement for the legacy ``download_soil_data.py``.

Usage (identical CLI to the legacy script)::

    # Download raw GeoTIFF files:
    python ag_cube_download_soil.py --config options/ag_cube_soil_config.yaml

    # Build the multi-depth datacube (set task: datacube in the YAML):
    python ag_cube_download_soil.py --config options/ag_cube_soil_config.yaml

What changed vs the legacy script
----------------------------------
| Legacy                          | New (ag-cube-cm)                          |
|---------------------------------|-------------------------------------------|
| ``SoilGridDataDonwload``        | ``SoilGridsDownloader``                   |
| ``get_soil_datacube``           | ``SoilDataCubeBuilder``                   |
| ``create_dimension``            | ``create_depth_dimension``                |
| ``set_encoding``                | ``set_encoding`` (raster_ops)             |
| ``check_crs_inxrdataset``       | ``check_crs_in_dataset`` (raster_ops)     |
| Raw OmegaConf dict surgery      | Clean builder API; no dict mutation        |

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
from ag_cube_cm.ingestion.soil import SoilGridsDownloader
from ag_cube_cm.transform.soil_cube import SoilDataCubeBuilder
from ag_cube_cm.spatial.raster_ops import (
    get_boundaries_from_shapefile,
    set_encoding,
    check_crs_in_dataset,
)

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
        description="Download and/or build SoilGrids soil datacubes "
                    "using the ag-cube-cm package."
    )
    parser.add_argument(
        "--config",
        default="options/ag_cube_soil_config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("ag-cube-cm soil pipeline starting …")
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    config = OmegaConf.load(args.config)
    logger.info("Configuration loaded from '%s'.", args.config)

    # -----------------------------------------------------------------------
    # Resolve spatial extent
    # -----------------------------------------------------------------------
    spatial_file = config.SPATIAL_INFO.get("boundaries", None)
    if spatial_file:
        extent = list(
            get_boundaries_from_shapefile(
                spatial_file,
                crs=config.SPATIAL_INFO.get("crs", None),
                round_numbers=True,
            )
        )
        logger.info("Extent derived from shapefile: %s → %s", spatial_file, extent)
    else:
        extent = list(config.SPATIAL_INFO.get("extent", []))
        logger.info("Extent from config: %s", extent)

    # Output folder:  <output_path>/<suffix>
    output_path: str = os.path.join(
        config.PATHS.output_path, config.GENERAL.suffix
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)

    soil_variables: list[str] = list(config.SOIL.variables)
    soil_depths: list[str]    = list(config.SOIL.depths)

    # -----------------------------------------------------------------------
    # TASK: download
    # Download raw GeoTIFF files from SoilGrids to disk.
    # -----------------------------------------------------------------------
    if config.GENERAL.task == "download":
        logger.info("Task: DOWNLOAD")

        downloader = SoilGridsDownloader(
            soil_layers=soil_variables,
            depths=soil_depths,
            output_folder=output_path,
        )
        written = downloader.download(boundaries=extent)
        logger.info("Download complete. %d files written.", len(written))
        for fp in written:
            logger.info("  → %s", fp)

    # -----------------------------------------------------------------------
    # TASK: datacube
    # Stack downloaded GeoTIFFs into a single multi-depth NetCDF datacube.
    # -----------------------------------------------------------------------
    elif config.GENERAL.task == "datacube":
        logger.info("Task: DATACUBE")

        builder = SoilDataCubeBuilder(
            data_folder=output_path,
            variables=soil_variables,
            extent=extent if extent else None,
            reference_variable="wv1500",          # spatial resolution reference
            crs="ESRI:54052",                      # SoilGrids native CRS
            target_crs="EPSG:4326",                # reproject output to WGS84
        )

        soil_cube = builder.build(verbose=True)

        # Ensure CRS metadata is consistent before saving
        soil_cube = check_crs_in_dataset(soil_cube)

        # Determine output filename
        datacube_fn = config.PATHS.get("datacube_fn", None)
        filename = (
            datacube_fn
            if datacube_fn
            else f"soil_data_{config.GENERAL.suffix}.nc"
        )
        out_nc = os.path.join(config.PATHS.output_path, filename)

        encoding = set_encoding(soil_cube)
        soil_cube.to_netcdf(out_nc, encoding=encoding, engine="netcdf4")
        logger.info("Soil datacube saved → %s", out_nc)

    else:
        raise ValueError(
            f"Unknown task '{config.GENERAL.task}'. "
            "Expected 'download' or 'datacube'."
        )


if __name__ == "__main__":
    main()
