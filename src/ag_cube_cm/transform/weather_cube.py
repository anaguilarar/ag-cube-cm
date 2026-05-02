"""
ag_cube_cm.transform.weather_cube
===================================

Transform Layer — Weather Datacube Builder.

**Rule:** This module transforms downloaded raw raster files into a single
multi-temporal NetCDF datacube suitable for crop model ingestion.
It does NOT download any data.

Key classes and functions
-------------------------
* :class:`MLTWeatherDataCube`  — reads per-variable, per-year folders of
  raw NetCDF files and stacks them into a single ``time × lat × lon`` cube.
* :func:`stack_datacube_temporally` — low-level stacking helper that
  concatenates a ``{date: xr.Dataset}`` dict along a ``time`` dimension
  while preserving the CRS.
* :func:`set_weather_encoding`  — zlib encoding dict for ``to_netcdf()``.

Chunking strategy
-----------------
Crop models extract data at individual pixel locations, so the optimal Dask
chunking strategy is to partition along **spatial** dimensions (lat/lon)
rather than time.  When building the final datacube we rechunk to::

    chunks = {"lat": 64, "lon": 64, "date": -1}   # all time steps per tile

This means each Dask task holds a small spatial tile but the full temporal
record for that tile, which is exactly what a pixel-level crop model loop
needs.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import xarray as xr

from ag_cube_cm.ingestion.files_manager import IntervalFolderManager  
from ag_cube_cm.ingestion.utils import set_crs 

logger = logging.getLogger(__name__)

# Standard variable name mapping (config key → cube variable name)
METEO_NAMES: dict[str, str] = {
    "precipitation": "precipitation",
    "solar_radiation": "srad",
    "temperature_tmax": "tmax",
    "temperature_tmin": "tmin",
    "vapour_pressure": "vp",
    "vapour_pressure_defficit": "vpd",
    "dew_point_temperature": "dpt",
    "wind_speed": "ws",
    "reference_evapotranspiration": "etr",
    "relative_humidity_06": "rh06",
    "relative_humidity_09": "rh09",
    "relative_humidity_12": "rh12",
    "relative_humidity_15": "rh15",
    "relative_humidity_18": "rh18",
}

# Optimal chunk sizes for pixel-level crop-model extraction
_SPATIAL_CHUNKS: dict[str, int | str] = {
    "date": -1,   # keep all time steps together
    "lat": 64,    # small spatial tiles
    "lon": 64,
    # also handle x/y dim names
    "y": 64,
    "x": 64,
}


# ---------------------------------------------------------------------------
# Low-level stacking helper
# ---------------------------------------------------------------------------


def stack_datacube_temporally(
    xrdata_dict: dict[str, xr.Dataset],
    time_dim_name: str = "date",
    parse_dates: bool = True,
) -> xr.Dataset:
    """Concatenate a per-date dict of datasets along a new temporal dimension.

    The CRS from the first dataset is preserved on the resulting datacube.

    Parameters
    ----------
    xrdata_dict : dict[str, xr.Dataset]
        Mapping of date strings (``"YYYYMMDD"``) to single-date datasets.
    time_dim_name : str
        Name for the new temporal coordinate.  Default: ``"date"``.
    parse_dates : bool
        If ``True``, parse the string keys as ``datetime`` objects using
        ``"%Y%m%d"`` format.  Default: ``True``.

    Returns
    -------
    xr.Dataset
        Multi-temporal dataset with CRS metadata preserved.
    """
    datasets: list[xr.Dataset] = []
    time_coords: list[Any] = (
        [datetime.strptime(k, "%Y%m%d") for k in xrdata_dict]
        if parse_dates
        else list(xrdata_dict.keys())
    )

    for t_coord, ds in zip(time_coords, xrdata_dict.values()):
        ds_exp = ds.assign_coords({time_dim_name: t_coord}).expand_dims(time_dim_name)
        datasets.append(ds_exp)

    stacked = xr.concat(datasets, dim=time_dim_name, combine_attrs="override", join='outer')

    # Restore CRS from first dataset
    first = datasets[0]
    try:
        import rioxarray  as rio 

        if first.rio.crs is not None:
            stacked.rio.write_crs(first.rio.crs, inplace=True)
        if first.rio.transform() is not None:
            stacked.rio.write_transform(first.rio.transform(), inplace=True)
    except Exception:  # noqa: BLE001
        pass

    if "spatial_ref" in stacked.variables:
        for var in stacked.data_vars:
            stacked[var].attrs["grid_mapping"] = "spatial_ref"

    stacked.attrs.update(first.attrs)
    logger.debug("Stacked %d slices along '%s'", len(datasets), time_dim_name)
    return stacked


def set_weather_encoding(
    xrdata: xr.Dataset, compress_method: str = "zlib"
) -> dict[str, dict]:
    """Build a zlib-compressed encoding dict for a weather datacube.

    Parameters
    ----------
    xrdata : xr.Dataset
        Dataset to encode.
    compress_method : str
        Compression method.  Default: ``"zlib"``.

    Returns
    -------
    dict[str, dict]
        Encoding dict for ``to_netcdf(encoding=...)``.
    """
    encoding: dict[str, dict] = {}
    for var in xrdata.data_vars:
        encoding[var] = {compress_method: True}
        if "grid_mapping" in xrdata[var].attrs:
            encoding[var]["grid_mapping"] = xrdata[var].attrs["grid_mapping"]
            del xrdata[var].attrs["grid_mapping"]
    if "spatial_ref" in xrdata.variables:
        encoding["spatial_ref"] = {}
    return encoding


# ---------------------------------------------------------------------------
# MLTWeatherDataCube
# ---------------------------------------------------------------------------


class MLTWeatherDataCube:
    """Build a multi-temporal weather NetCDF datacube from raw per-variable files.

    Parameters
    ----------
    directory_paths : dict[str, str]
        Mapping of ``{variable_name: folder_path}`` for each weather variable.
        Example::

            {
                "tmax": "data/raw/temperature_tmax_mwi_raw",
                "tmin": "data/raw/temperature_tmin_mwi_raw",
                "precipitation": "data/raw/precipitation_mwi_raw",
            }
    folder_manager : IntervalFolderManager
        Legacy folder manager used to discover and index date-file pairs.
    extent : list[float] | None
        Optional spatial clipping extent ``[xmin, ymin, xmax, ymax]``.

    Examples
    --------
    >>> from ag_cube_cm.ingestion.files_manager import IntervalFolderManager
    >>> cube = MLTWeatherDataCube(
    ...     directory_paths={"tmax": "data/raw/tmax", "tmin": "data/raw/tmin"},
    ...     folder_manager=IntervalFolderManager(),
    ... )
    >>> cube.common_dates_and_file_names("2010-01-01", "2019-12-31")
    >>> mlt = cube.multitemporal_data(reference_variable="tmax")
    >>> stacked = stack_datacube_temporally(mlt)
    """

    def __init__(
        self,
        directory_paths: dict[str, str],
        folder_manager: IntervalFolderManager,
        extent: list[float] | None = None,
    ) -> None:
        self.directory_paths = directory_paths
        self.folder_manager = folder_manager
        self._extent = extent
        self.available_dates: dict[str, list[str]] = {}
        self.available_files: dict[str, list[str]] = {}
        self._query_dates: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def variables(self) -> list[str]:
        """List of variable names registered in ``directory_paths``."""
        return list(self.directory_paths.keys())

    def common_dates_and_file_names(
        self,
        starting_date: str,
        ending_date: str,
    ) -> dict[str, dict[str, str]]:
        """Find the intersection of available dates across all variables.

        Parameters
        ----------
        starting_date : str
            Start date ``"YYYY-MM-DD"``.
        ending_date : str
            End date ``"YYYY-MM-DD"``.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{date_str: {variable: file_path}}`` for all common dates.
        """
        all_dates: list[list[str]] = []
        all_files: list[list[str]] = []

        for var in self.variables:
            dates, files = self._get_date_paths(
                var, starting_date=starting_date, ending_date=ending_date
            )
            all_dates.append(dates)
            all_files.append(files)

        common_dates, filtered_files = self._filter_common_dates(all_dates, all_files)

        self._query_dates = {
            d: {self.variables[j]: filtered_files[j][i] for j in range(len(self.variables))}
            for i, d in enumerate(common_dates)
        }
        logger.info(
            "Found %d common dates across %d variables.",
            len(common_dates), len(self.variables),
        )
        return self._query_dates

    def multitemporal_data(
        self,
        reference_variable: str = "precipitation",
        ncores: int = 0,
        **kwargs: Any,
    ) -> dict[str, xr.Dataset]:
        """Build a per-date dict of multi-variable datasets.

        Parameters
        ----------
        reference_variable : str
            Variable whose grid is used as the spatial reference for
            co-registration.  Default: ``"precipitation"``.
        ncores : int
            Parallel workers.  ``0`` → sequential.  Default: 0.

        Returns
        -------
        dict[str, xr.Dataset]
            ``{date_str: xr.Dataset}`` with all variables merged.
        """
        if not self._query_dates:
            raise RuntimeError(
                "Call common_dates_and_file_names() before multitemporal_data()."
            )

        if ncores > 0:
            return self._build_parallel(reference_variable, ncores, **kwargs)
        return self._build_sequential(reference_variable, **kwargs)

    def save_datacube(
        self,
        output_path: str,
        starting_date: str,
        ending_date: str,
        suffix: str = "",
        reference_variable: str = "tmax",
        rechunk_for_spatial: bool = True,
        ncores: int = 0,
    ) -> str:
        """Build and save the full multi-temporal datacube to a NetCDF file.

        This is the high-level convenience method that orchestrates the full
        pipeline: discover dates → build per-date datasets → stack temporally
        → rechunk for pixel extraction → save with zlib encoding.

        Parameters
        ----------
        output_path : str
            Directory where the output ``.nc`` file will be saved.
        starting_date : str
            Start date ``"YYYY-MM-DD"``.
        ending_date : str
            End date ``"YYYY-MM-DD"``.
        suffix : str
            Optional name suffix for the output file.
        reference_variable : str
            Spatial reference variable for co-registration.
        rechunk_for_spatial : bool
            If ``True``, rechunk the final cube for pixel-level extraction
            (spatial tiles, full time axis).  Default: ``True``.
        ncores : int
            Parallel workers for per-date processing.  ``0`` → sequential.
            Default: 0.

        Returns
        -------
        str
            Path to the saved NetCDF file.
        """
        self.common_dates_and_file_names(starting_date, ending_date)
        mlt = self.multitemporal_data(reference_variable=reference_variable, ncores=ncores)
        cube = stack_datacube_temporally(mlt, time_dim_name="date", parse_dates=True)

        if rechunk_for_spatial:
            # Rechunk: all time steps per spatial tile — optimal for pixel loops
            valid_dims = {k: v for k, v in _SPATIAL_CHUNKS.items() if k in cube.dims}
            if valid_dims:
                cube = cube.chunk(valid_dims)
                logger.debug("Rechunked for spatial extraction: %s", valid_dims)

        sy = starting_date[:4]
        ey = ending_date[:4]
        fname = f"weather_{suffix}_{sy}_{ey}.nc" if suffix else f"weather_{sy}_{ey}.nc"
        out_file = os.path.join(output_path, fname)

        encoding = set_weather_encoding(cube)
        cube.to_netcdf(out_file, encoding=encoding, engine="netcdf4")
        logger.info("Weather datacube saved → %s", out_file)
        return out_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_date(self, date: str) -> dict[str, str]:
        """Return ``{variable: absolute_file_path}`` for a single date."""
        year = date[:4]
        file_map = self._query_dates[date]
        return {
            var: os.path.join(self.directory_paths[var], year, fname)
            for var, fname in file_map.items()
        }

    def _get_date_paths(
        self, variable: str, starting_date: str, ending_date: str
    ) -> tuple[list[str], list[str]]:
        """Discover available date-file pairs for a single variable."""
        result = self.folder_manager(
            self.directory_paths[variable],
            starting_date=starting_date,
            ending_date=ending_date,
        )
        dates_arr, files_arr = np.array(result).T
        self.available_dates[variable] = dates_arr.tolist()
        self.available_files[variable] = files_arr.tolist()
        return self.available_dates[variable], self.available_files[variable]

    @staticmethod
    def _filter_common_dates(
        dates: list[list[str]], filenames: list[list[str]]
    ) -> tuple[list[str], list[list[str]]]:
        """Intersect date lists across variables and filter file lists accordingly."""
        common = sorted(set(dates[0]).intersection(*dates[1:]))
        filtered = [
            [filenames[j][i] for i, d in enumerate(dates[j]) if d in common]
            for j in range(len(dates))
        ]
        return common, filtered

    def _stack_single_date(
        self, date: str, reference_variable: str, **kwargs: Any
    ) -> xr.Dataset:
        """Load and co-register all variables for a single date."""
        from ag_cube_cm.ingestion.utils import resample_variables  # legacy
        from ag_cube_cm.ingestion.gis_functions import read_raster_data  # legacy

        paths = self._query_date(date)
        var_datasets: dict[str, xr.Dataset] = {}
        for var, fp in paths.items():
            var_datasets[var] = read_raster_data(fp, crop_extent=self._extent)

        return resample_variables(
            var_datasets, reference_variable=reference_variable, **kwargs
        )

    def _build_sequential(
        self, reference_variable: str, **kwargs: Any
    ) -> dict[str, xr.Dataset]:
        """Build the per-date dict sequentially with a progress bar."""
        xr_dict: dict[str, xr.Dataset] = {}
        for date in tqdm.tqdm(self._query_dates, desc="Building weather cube"):
            ds = self._stack_single_date(date, reference_variable, **kwargs)
            ds = set_crs(ds, ds.attrs.get("crs") or ds.rio.crs)
            xr_dict[date] = ds
        return xr_dict

    def _build_parallel(
        self, reference_variable: str, ncores: int, **kwargs: Any
    ) -> dict[str, xr.Dataset]:
        """Build the per-date dict in parallel using ProcessPoolExecutor."""
        xr_dict: dict[str, xr.Dataset] = {}
        dates = list(self._query_dates.keys())

        with tqdm.tqdm(total=len(dates), desc="Building weather cube (parallel)") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as pool:
                future_map = {
                    pool.submit(
                        self._stack_single_date, d, reference_variable, **kwargs
                    ): d
                    for d in dates
                }
                for future in concurrent.futures.as_completed(future_map):
                    date = future_map[future]
                    try:
                        xr_dict[date] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Date %s failed: %s", date, exc)
                    pbar.update(1)

        return xr_dict
