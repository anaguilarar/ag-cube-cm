"""
ag_cube_cm.ingestion.weather
=============================

Data Ingestion Layer — Weather.

**Rule:** This module ONLY downloads files to disk.
It does NOT build data cubes, transform units, or open datasets into memory.

Two downloader classes are provided:

* :class:`AgEra5Downloader`  — wraps the Copernicus CDS API for all AgEra5
  agrometeorological indicator variables.
* :class:`CHIRPSDownloader` — downloads CHIRPS daily precipitation from the
  UCSB data warehouse via direct COG/GeoTIFF URLs.

Both classes follow the same interface contract:
``download(extent, starting_date, ending_date, output_folder, ncores) -> dict[str, str]``
where the returned dict maps year strings to output file/folder paths.

The higher-level :class:`WeatherDownloadOrchestrator` mirrors the legacy
``ClimateDataDownload`` class so existing workflow scripts keep working with
minimal changes.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio
import xarray
from rasterio.mask import mask as rio_mask
from shapely.geometry import Polygon

from ag_cube_cm.ingestion.files_manager import (
    create_yearly_query,
    days_range_asstring,
    find_date_instring,
    months_range_asstring,
    split_date,
    uncompress_zip_path,
)
from ag_cube_cm.ingestion.gis_functions import (
    from_polygon_2bbox,
    from_xyxy_2polygon,
    numpy_to_xarray,
    read_raster_data,
)
from ag_cube_cm.ingestion.utils import download_file  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Date / query helpers (thin wrappers kept here for internal use)
# ---------------------------------------------------------------------------



def process_file(year_path_folder:str, filename:str, date:str, xdim_name:str, ydim_name:str, depthdim_name:str):
    """Reads and processes a single NetCDF file into an xarray.Dataset.

    Parameters
    ----------
    year_path_folder : str
        The path to the folder containing the file.
    filename : str
        The name of the NetCDF file.
    date : str
        The date string ('YYYYMMDD') associated with the file.
    xdim_name : str
        The name of the x-dimension (e.g., 'longitude').
    ydim_name : str
        The name of the y-dimension (e.g., 'latitude').
    depthdim_name : str
        The name for the new time dimension.

    Returns
    -------
    xarray.Dataset
        A dataset containing the data from the file with a new time dimension.
    """
    dateasdatetime = datetime.strptime(date, '%Y%m%d')
    filepath = os.path.join(year_path_folder, filename)
    xrdata = read_raster_data(filepath, ydim_name=ydim_name, xdim_name=xdim_name)
    
    # Assuming only one variable in the file
    varname = list(xrdata.data_vars.keys())[0]

    # Reorganize data
    two_var = xrdata[varname].values[0] if len(xrdata[varname].values.shape) == 3 else xrdata[varname].values
    xrdata = xarray.Dataset(
        data_vars={str(dateasdatetime.year): ([ydim_name, xdim_name], two_var)},
        coords={xdim_name: xrdata[xdim_name].values, ydim_name: xrdata[ydim_name].values}
    )
    
    # Add a time dimension
    xrdata = xrdata.expand_dims(dim={depthdim_name: 1}, axis=0)
    xrdata[depthdim_name] = [dateasdatetime]

    return xrdata

def read_annual_data(path: str, year:str,xdim_name: str = 'longitude',
                        ydim_name: str = 'latitude',
                        depthdim_name: str = 'time',
                        crs: str = 'EPSG:4326'):
    """
    Reads annual data from NetCDF files for a given year and compiles it into a multi-temporal xarray Dataset.

    Parameters:
    -----------
    path : str
        The folder path containing the yearly data.
    year : str
        The year for which data needs to be read.
    xdim_name : str, optional
        Name of the x-dimension (longitude), default is 'longitude'.
    ydim_name : str, optional
        Name of the y-dimension (latitude), default is 'latitude'.
    depthdim_name : str, optional
        Name of the depth dimension (time), default is 'time'.
    crs : str, optional
        Coordinate Reference System (CRS), default is 'EPSG:4326'.
    
    Returns:
    --------
    annual_data : xarray.Dataset
        A concatenated Dataset with time as one of the dimensions.
    """
        
    # find folder path
    year_path_folder = uncompress_zip_path(path, year)
    #get dates and filenmaes with the extension
    times = ([[fn, find_date_instring(fn, pattern=year)] for fn in os.listdir(year_path_folder) if fn.endswith('.nc')])
    
    #read data
    list_xrdata = [process_file(os.path.join(path, year), fn, date, 
                                ydim_name, xdim_name, depthdim_name) for fn, date in times]
    
    annual_data = xarray.concat(list_xrdata, dim = depthdim_name)

    import rioxarray  # noqa: F401  — required before any .rio accessor call
    tmp = list_xrdata[0].copy().rio.write_crs(crs)
    spatial_ref = tmp.rio.write_transform(tmp.rio.transform()).spatial_ref
    annual_data =  annual_data.assign(crs = spatial_ref)

    if 'spatial_ref' in list(annual_data.coords.keys()):
        return annual_data.drop_vars('spatial_ref')
    else:
        return annual_data


def _transform_dates_for_agera5_query(
    year: int,
    init_day: int | None = None,
    end_day: int | None = None,
    init_month: int | None = None,
    end_month: int | None = None,
) -> dict[str, list[str]]:
    """Build the ``year / month / day`` sub-dict for a CDS API request.

    Parameters
    ----------
    year : int
        The year for the query.
    init_day : int, optional
        First day of the month (1–31). Defaults to 1.
    end_day : int, optional
        Last day of the month (1–31). Defaults to 31.
    init_month : int, optional
        First month (1–12). Defaults to 1.
    end_month : int, optional
        Last month (1–12). Defaults to 12.

    Returns
    -------
    dict[str, list[str]]
        ``{'year': ['YYYY'], 'month': ['MM', ...], 'day': ['DD', ...]}``
    """
    init_day = init_day or 1
    init_month = init_month or 1
    end_month = end_month or 12
    end_day = end_day or 31

    return {
        "year": [str(year)],
        "month": months_range_asstring(init_month, end_month),
        "day": days_range_asstring(init_day, end_day),
    }


# ---------------------------------------------------------------------------
# AgEra5 downloader
# ---------------------------------------------------------------------------


class AgEra5Downloader:
    """Download AgEra5 agrometeorological data from the Copernicus CDS API.

    Each call to :meth:`download` submits one CDS API request per year,
    optionally in parallel (``ncores > 0``), and saves the raw ``.zip``
    files to ``output_folder``.

    Parameters
    ----------
    product : str
        CDS dataset identifier.  Defaults to
        ``"sis-agrometeorological-indicators"``.
    version : str
        AgEra5 product version.  ``"2_0"`` (default) or ``"1_1"``.
    max_attempts : int
        Maximum retry attempts per year.  Defaults to 3.

    Examples
    --------
    >>> dl = AgEra5Downloader()
    >>> paths = dl.download(
    ...     variable="2m_temperature",
    ...     statistic=["24_hour_maximum"],
    ...     starting_date="2010-01-01",
    ...     ending_date="2010-12-31",
    ...     output_folder="data/raw/tmax",
    ...     aoi_extent=[-10.5, 35.0, -10.0, 36.0],
    ...     ncores=4,
    ... )
    """

    #: Default CDS dataset identifier for AgEra5.
    PRODUCT: str = "sis-agrometeorological-indicators"

    def __init__(
        self,
        product: str = "sis-agrometeorological-indicators",
        version: str = "2_0",
        max_attempts: int = 3,
    ) -> None:
        self.product = product
        self.version = version
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        variable: str | list[str],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        aoi_extent: list[float],
        statistic: list[str] | None = None,
        time: list[str] | None = None,
        ncores: int = 4,
    ) -> dict[str, str]:
        """Download AgEra5 data for a variable and date range.

        Parameters
        ----------
        variable : str | list[str]
            AgEra5 variable name(s), e.g. ``"2m_temperature"``.
        starting_date : str
            ISO 8601 start date ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end date ``"YYYY-MM-DD"``.
        output_folder : str
            Root directory; one ``.zip`` file is written per year.
        aoi_extent : list[float]
            Bounding box ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
            Internally re-ordered to CDS format ``[N, W, S, E]``.
        statistic : list[str] | None
            Statistic filter, e.g. ``["24_hour_maximum"]``.
        time : list[str] | None
            Hour filter for relative-humidity (e.g. ``["12_00"]``).
        ncores : int
            Parallel threads.  ``0`` → sequential.  Default: 4.

        Returns
        -------
        dict[str, str]
            ``{year_str: zip_file_path}``
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        init_year, init_month, init_day = split_date(starting_date)
        end_year, end_month, end_day = split_date(ending_date)
        years = list(range(init_year, end_year + 1))

        # CDS expects [N, W, S, E]
        cds_area = [aoi_extent[3], aoi_extent[0], aoi_extent[1], aoi_extent[2]]

        base_query: dict[str, Any] = {
            "version": self.version,
            "area": cds_area,
            "variable": variable if isinstance(variable, list) else [variable],
        }
        if statistic:
            base_query["statistic"] = statistic
        if time:
            base_query["time"] = time

        if ncores > 0:
            file_path_per_year = self._parallel_download(
                years, base_query, output_folder,
                init_year, end_year, init_month, end_month, init_day, end_day,
                ncores,
            )
        else:
            file_path_per_year = self._sequential_download(
                years, base_query, output_folder,
                init_year, end_year, init_month, end_month, init_day, end_day,
            )

        return file_path_per_year

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_one_year(
        self,
        year: int,
        base_query: dict[str, Any],
        output_folder: str,
        init_year: int,
        end_year: int,
        init_month: int,
        end_month: int,
        init_day: int,
        end_day: int,
    ) -> str | None:
        """Issue a single-year CDS request and return the output zip path."""
        try:
            import cdsapi  # lazy import — optional dependency
        except ImportError as exc:
            raise ImportError(
                "cdsapi is required for AgEra5 downloads. "
                "Install it with: pip install cdsapi"
            ) from exc

        try:
            if year == init_year:
                dates = _transform_dates_for_agera5_query(
                    year, init_day=init_day, end_day=31,
                    init_month=init_month, end_month=12,
                )
            elif year == end_year:
                dates = _transform_dates_for_agera5_query(
                    year, init_day=1, end_day=end_day,
                    init_month=1, end_month=end_month,
                )
            else:
                dates = _transform_dates_for_agera5_query(year)

            query = copy.deepcopy(base_query)
            query.update(dates)
            zip_path = os.path.join(output_folder, f"{year}.zip")

            logger.info("CDS request: year=%d  query=%s", year, query)
            client = cdsapi.Client()
            client.retrieve(self.product, query, zip_path)
            logger.info("Downloaded year=%d → %s", year, zip_path)
            return zip_path

        except Exception as exc:  # noqa: BLE001
            logger.warning("Year %d download failed: %s", year, exc)
            return None

    def _sequential_download(
        self,
        years: list[int],
        base_query: dict[str, Any],
        output_folder: str,
        init_year: int,
        end_year: int,
        init_month: int,
        end_month: int,
        init_day: int,
        end_day: int,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for year in years:
            for attempt in range(1, self.max_attempts + 1):
                logger.info("Year %d – attempt %d/%d", year, attempt, self.max_attempts)
                path = self._download_one_year(
                    year, base_query, output_folder,
                    init_year, end_year, init_month, end_month, init_day, end_day,
                )
                if path:
                    results[str(year)] = path
                    break
            else:
                logger.error("Year %d failed after %d attempts.", year, self.max_attempts)
        return results

    def _parallel_download(
        self,
        years: list[int],
        base_query: dict[str, Any],
        output_folder: str,
        init_year: int,
        end_year: int,
        init_month: int,
        end_month: int,
        init_day: int,
        end_day: int,
        ncores: int,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        tasks_to_retry: dict[int, int] = {y: 1 for y in years}

        while tasks_to_retry:
            this_round = tasks_to_retry.copy()
            tasks_to_retry.clear()

            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
                future_map = {
                    pool.submit(
                        self._download_one_year, y, base_query, output_folder,
                        init_year, end_year, init_month, end_month, init_day, end_day,
                    ): y
                    for y in this_round
                }
                for future in concurrent.futures.as_completed(future_map):
                    year = future_map[future]
                    attempt = this_round[year]
                    try:
                        path = future.result()
                        if path:
                            results[str(year)] = path
                        else:
                            raise RuntimeError("download returned None")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Year %d attempt %d failed: %s", year, attempt, exc)
                        if attempt < self.max_attempts:
                            tasks_to_retry[year] = attempt + 1
                        else:
                            logger.error(
                                "Year %d gave up after %d attempts.", year, self.max_attempts
                            )
        return results

    # ------------------------------------------------------------------
    # Post-download stacking helper (kept for backwards compat)
    # ------------------------------------------------------------------

    @staticmethod
    def stack_annual_to_netcdf(
        raw_folder: str,
        init_year: int,
        end_year: int,
        output_folder: str | None = None,
        remove_source: bool = True,
    ) -> None:
        """Stack per-day NetCDF files within a year into a single annual file.

        Parameters
        ----------
        raw_folder : str
            Directory containing the downloaded ``.zip`` / unzipped folders.
        init_year : int
            First year to stack.
        end_year : int
            Last year to stack (inclusive).
        output_folder : str | None
            Destination for the stacked ``.nc`` files.  Defaults to
            ``raw_folder``.
        remove_source : bool
            Delete the original zip and unzipped folder after stacking.
        """

        out = output_folder or raw_folder
        for year in range(init_year, end_year + 1):
            try:
                ds = read_annual_data(raw_folder, str(year))
                nc_path = os.path.join(out, f"{year}.nc")
                ds.to_netcdf(nc_path)
                logger.info("Stacked year %d → %s", year, nc_path)
                if remove_source:
                    yr_dir = os.path.join(raw_folder, str(year))
                    yr_zip = yr_dir + ".zip"
                    if os.path.isdir(yr_dir):
                        shutil.rmtree(yr_dir)
                    if os.path.isfile(yr_zip):
                        os.remove(yr_zip)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not stack year %d: %s", year, exc)


# ---------------------------------------------------------------------------
# CHIRPS downloader
# ---------------------------------------------------------------------------


class CHIRPSDownloader:
    """Download CHIRPS daily precipitation data from the UCSB servers.

    Files are read directly from the cloud-optimised GeoTIFF (COG) endpoints
    using ``rasterio``, clipped to the AOI, and saved as daily NetCDF files.

    Parallelism is at the **year** level: one thread downloads all days of a
    given year sequentially.  This matches the legacy ``download_data_per_year``
    pattern and avoids triggering server-side rate limits.

    Parameters
    ----------
    frequency : str
        ``"daily"`` (default) or ``"monthly"``.
    sp_resolution : str
        Spatial resolution code — ``"05"`` for 0.05°.  Default ``"05"``.
    version : str
        CHIRPS version: ``"3.0"`` (default) or ``"2.0"``.

    Examples
    --------
    >>> dl = CHIRPSDownloader()
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2018-01-01",
    ...     ending_date="2018-12-31",
    ...     output_folder="data/raw/precipitation",
    ...     ncores=4,
    ... )
    """

    #: URL template for CHIRPS 2.0
    URL_V2: str = (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
        "global_{freq}/cogs/p{res}/{year}/chirps-v2.0.{date}.cog"
    )
    #: URL template for CHIRP v3.0
    URL_V3: str = (
        "https://data.chc.ucsb.edu/products/CHIRP-v3.0/"
        "{freq}/global/tifs/{year}/chirp-v3.0.{date}.tif"
    )

    def __init__(
        self,
        frequency: str = "daily",
        sp_resolution: str = "05",
        version: str = "3.0",
    ) -> None:
        self.frequency = frequency
        self.resolution = sp_resolution
        self.version = version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        ncores: int = 6,
        polite_delay: float = 0.1,
    ) -> dict[str, str]:
        """Download CHIRPS data for the given extent and date range.

        All (year, month, day) jobs are flattened into a single queue and
        processed by a ``ThreadPoolExecutor`` with ``ncores`` workers.  This
        gives true day-level parallelism across years while keeping the total
        number of concurrent HTTP connections low enough to avoid CrowdSec
        rate-limiting on ``data.chc.ucsb.edu``.

        Each worker sleeps ``polite_delay`` seconds after every completed
        request to stagger bursts and stay within polite request-rate limits.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` bounding box in EPSG:4326.
        starting_date : str
            ISO 8601 start date ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end date ``"YYYY-MM-DD"``.
        output_folder : str
            Root folder; one sub-folder per year is created automatically.
        ncores : int
            Maximum concurrent day-downloads.  Default: 6.  Keep ≤ 8 to
            avoid triggering server-side rate limits.  ``0`` → sequential.
        polite_delay : float
            Seconds each worker sleeps after completing a request.
            Default: 0.1 s.  Set to 0 to disable.

        Returns
        -------
        dict[str, str]
            ``{year_str: year_folder_path}``
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        yearly_dates = create_yearly_query(starting_date, ending_date)

        # Pre-create year folders and build flat job list.
        year_folders: dict[str, str] = {}
        jobs: list[tuple[str, str, str]] = []
        for year, monthly_dates in yearly_dates.items():
            year_folder = os.path.join(output_folder, year)
            Path(year_folder).mkdir(parents=True, exist_ok=True)
            year_folders[year] = year_folder
            for month, days in monthly_dates.items():
                for day in days:
                    jobs.append((year, month, day))

        pbar = tqdm(total=len(jobs), desc="CHIRPS precipitation", unit="day")

        def _run(job: tuple[str, str, str]) -> None:
            year, month, day = job
            try:
                self._download_one_day(year, month, day, output_folder, extent)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed %s-%s-%s: %s", year, month, day, exc)
            finally:
                pbar.update(1)
            if polite_delay > 0:
                time.sleep(polite_delay)

        if ncores > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
                list(pool.map(_run, jobs))
        else:
            for job in jobs:
                _run(job)

        pbar.close()
        return year_folders

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_one_year(
        self,
        year: str,
        monthly_dates: dict,
        output_folder: str,
        extent: list[float],
        pbar: tqdm,
    ) -> None:
        """Download all days of *year* sequentially, updating *pbar* per day."""
        for month, days in monthly_dates.items():
            for day in days:
                try:
                    self._download_one_day(year, month, day, output_folder, extent)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed %s-%s-%s: %s", year, month, day, exc)
                pbar.update(1)

    def _build_url(self, year: str, date_str: str, version: str | None = None) -> str:
        """Return the full download URL for a single daily file."""
        if version is None:
            version = self.version
        # CHIRP v3.0 only covers 2000-onward; fall back to v2.0 for earlier years.
        if version == "3.0" and int(year) <= 2000:
            version = "2.0"

        if version == "2.0":
            return self.URL_V2.format(
                freq=self.frequency, res=self.resolution, year=year, date=date_str
            )
        return self.URL_V3.format(
            freq=self.frequency, year=year, date=date_str
        )

    def _open_url(self, url: str) -> rasterio.DatasetReader:
        """Open a rasterio dataset, raising the original exception on failure."""
        return rasterio.open(url)

    def _download_one_day(
        self,
        year: str,
        month: str,
        day: str,
        output_folder: str,
        extent: list[float],
    ) -> None:
        """Download and clip a single daily CHIRPS file to NetCDF.

        Tries the configured version first; if the server returns an HTTP error
        (e.g. 403 on CHIRP v3.0) it automatically retries with CHIRPS v2.0.
        """
        date_str = f"{year}.{month}.{day}"
        out_nc = os.path.join(
            output_folder, year, f"chirps_precipitation_{year}{month}{day}.nc"
        )
        if os.path.exists(out_nc):
            return  # skip if already downloaded

        # Build candidate URLs: primary version first, then v2.0 fallback.
        primary_url = self._build_url(year, date_str)
        fallback_url = (
            self._build_url(year, date_str, version="2.0")
            if self.version != "2.0" else None
        )
        urls = [primary_url] + ([fallback_url] if fallback_url else [])

        last_exc: Exception | None = None
        for url in urls:
            try:
                with rasterio.open(url) as src:
                    aoi_geom = gpd.GeoSeries([from_xyxy_2polygon(*extent)])
                    masked, transform = rio_mask(
                        dataset=src,
                        shapes=aoi_geom,
                        crop=True,
                    )
                    if masked.shape[0] > 1:
                        masked = np.expand_dims(masked[-1], axis=0)
                    xrm = numpy_to_xarray(
                        masked, transform, crs=str(src.crs), var_name="precipitation"
                    )
                    xrm.to_netcdf(out_nc)
                    if url != primary_url:
                        logger.debug("Used fallback URL for %s: %s", date_str, url)
                    else:
                        logger.debug("Saved %s", out_nc)
                return  # success — stop trying
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("URL failed (%s): %s", url, exc)

        logger.warning("Failed to download %s: %s", primary_url, last_exc)


# ---------------------------------------------------------------------------
# High-level orchestrator (mirrors legacy ClimateDataDownload)
# ---------------------------------------------------------------------------

#: Maps config variable keys to AgEra5 API variable names + statistics.
_AGERA5_VARIABLE_MAP: dict[str, dict] = {
    "wind_speed": {
        "variable": "10m_wind_speed",
        "statistic": ["24_hour_mean"],
    },
    "vapour_pressure": {
        "variable": "vapour_pressure",
        "statistic": ["24_hour_mean"],
    },
    "vapour_pressure_defficit": {
        "variable": "vapour_pressure_deficit_at_maximum_temperature",
        "statistic": ["24_hour_mean"],
    },
    "relative_humidity_max": {
        "variable": "2m_relative_humidity_derived",
        "statistic": ["24_hour_maximum"],
    },
    "relative_humidity_min": {
        "variable": "2m_relative_humidity_derived",
        "statistic": ["24_hour_minimum"],
    },
    "relative_humidity_06": {"variable": "2m_relative_humidity", "time": ["06_00"]},
    "relative_humidity_09": {"variable": "2m_relative_humidity", "time": ["09_00"]},
    "relative_humidity_12": {"variable": "2m_relative_humidity", "time": ["12_00"]},
    "relative_humidity_15": {"variable": "2m_relative_humidity", "time": ["15_00"]},
    "relative_humidity_18": {"variable": "2m_relative_humidity", "time": ["18_00"]},
    "reference_evapotranspiration": {"variable": "reference_evapotranspiration"},
    "solar_radiation": {"variable": "solar_radiation_flux"},
    "dew_point_temperature": {
        "variable": "2m_dewpoint_temperature",
        "statistic": ["24_hour_mean"],
    },
    "temperature_tmax": {
        "variable": "2m_temperature",
        "statistic": ["24_hour_maximum"],
    },
    "temperature_tmin": {
        "variable": "2m_temperature",
        "statistic": ["24_hour_minimum"],
    },
}


class WeatherDownloadOrchestrator:
    """Orchestrate multi-variable weather data downloads.

    Replaces the legacy ``ClimateDataDownload`` class with cleaner separation:
    CHIRPS → :class:`CHIRPSDownloader`, AgEra5 → :class:`AgEra5Downloader`.

    Parameters
    ----------
    starting_date : str
        Start date ``"YYYY-MM-DD"``.
    ending_date : str
        End date ``"YYYY-MM-DD"``.
    xyxy : list[float]
        Bounding box ``[xmin, ymin, xmax, ymax]``.
    output_folder : str
        Root folder for all downloads.
    aoi : Polygon | None
        Alternative to ``xyxy`` — bounding box derived from this polygon.

    Examples
    --------
    >>> orch = WeatherDownloadOrchestrator(
    ...     starting_date="2010-01-01",
    ...     ending_date="2019-12-31",
    ...     xyxy=[-90.5, 13.0, -88.5, 15.5],
    ...     output_folder="data/raw",
    ... )
    >>> orch.download({"temperature_tmax": {"mission": "agera5", "source": "agera5"}})
    """

    def __init__(
        self,
        starting_date: str,
        ending_date: str,
        xyxy: list[float] | None = None,
        output_folder: str | None = None,
        aoi: Polygon | None = None,
    ) -> None:
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.extent: list[float] = from_polygon_2bbox(aoi) if aoi else (xyxy or [])
        self.output_folder = output_folder
        if output_folder:
            Path(output_folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        weather_variables: dict[str, dict],
        suffix: str | None = None,
        export_as_netcdf: bool = False,
        ncores: int = 4,
        version: str = "2_0",
    ) -> dict[str, dict[str, str]]:
        """Download all requested weather variables.

        Parameters
        ----------
        weather_variables : dict[str, dict]
            Keys are variable names (e.g. ``"temperature_tmax"``); values are
            dicts with ``mission`` and ``source`` keys.
        suffix : str | None
            Optional folder suffix (e.g. region name).
        export_as_netcdf : bool
            If ``True``, stack yearly zip files into ``.nc`` after download.
        ncores : int
            Parallel workers.
        version : str
            AgEra5 product version.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{variable: {year: path}}``
        """
        results: dict[str, dict[str, str]] = {}

        for var_key, info in weather_variables.items():
            out_folder = self._make_output_folder(var_key, suffix)
            mission = info.get("mission", "agera5")
            source = info.get("source", "agera5")

            matched_key = next(
                (k for k in _AGERA5_VARIABLE_MAP if k in var_key), None
            )

            if var_key == "precipitation" or (
                matched_key is None and "precipitation" in var_key
            ):
                # CHIRPS path
                chirps = CHIRPSDownloader()
                paths = chirps.download(
                    extent=self.extent,
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                    ncores=ncores,
                )
                results[var_key] = paths

            elif matched_key and mission == "agera5" and source == "agera5":
                # AgEra5 path
                spec = _AGERA5_VARIABLE_MAP[matched_key]
                agera5 = AgEra5Downloader(version=version)
                paths = agera5.download(
                    variable=spec["variable"],
                    statistic=spec.get("statistic"),
                    time=spec.get("time"),
                    starting_date=self.starting_date,
                    ending_date=self.ending_date,
                    output_folder=out_folder,
                    aoi_extent=self.extent,
                    ncores=ncores,
                )
                results[var_key] = paths

                if export_as_netcdf and paths:
                    years = sorted(int(y) for y in paths)
                    AgEra5Downloader.stack_annual_to_netcdf(
                        out_folder, years[0], years[-1], out_folder
                    )
            else:
                logger.warning("Variable '%s' is not yet implemented.", var_key)
                results[var_key] = {}

        return results

    def _make_output_folder(self, variable: str, suffix: str | None) -> str:
        name = f"{variable}_{suffix}_raw" if suffix else f"{variable}_raw"
        path = os.path.join(self.output_folder or ".", name)
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
