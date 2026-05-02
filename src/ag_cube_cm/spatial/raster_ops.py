"""
ag_cube_cm.spatial.raster_ops
==============================

Standalone spatial / raster operation functions extracted from the legacy
``spatialdata.gis_functions`` and ``spatialdata.utils`` modules.

All functions in this module are **pure** (no side-effects) and operate on
lazy ``xr.Dataset`` / ``xr.DataArray`` objects unless otherwise noted.
They do not trigger Dask computation unless explicitly required.

Key functions
-------------
* :func:`clip_to_bbox`          — clip a dataset to a bounding box.
* :func:`reproject_dataset`     — reproject to a target CRS via rioxarray.
* :func:`mask_with_geometry`    — mask/clip using a GeoDataFrame geometry.
* :func:`get_roi_data`          — full Region-Of-Interest extraction with
  optional geometry masking; CRS resolution is ``@lru_cache``-cached.
* :func:`set_encoding`          — build NetCDF zlib encoding dict.
* :func:`check_crs_in_dataset`  — ensure ``spatial_ref`` / CRS metadata is
  consistently written before saving.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRS helpers (cached to avoid repeated pyproj overhead in pixel loops)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _resolve_crs(crs_string: str) -> Any:
    """Parse and cache a CRS string using pyproj.

    The ``@lru_cache`` means that the first call for a given ``crs_string``
    pays the pyproj parsing cost; subsequent calls return the cached object
    instantly.  This is important in pixel-level loops where the same CRS is
    resolved thousands of times.

    Parameters
    ----------
    crs_string : str
        Any CRS string understood by pyproj (EPSG, WKT, PROJ string, …).

    Returns
    -------
    pyproj.CRS
        Parsed CRS object.
    """
    from pyproj import CRS

    return CRS.from_user_input(crs_string)


def _get_dataset_crs_string(ds: xr.Dataset) -> str | None:
    """Extract the CRS string from a dataset's rioxarray accessor.

    Returns ``None`` if no CRS can be found, rather than raising.
    """
    try:
        import rioxarray  # noqa: F401

        crs = ds.rio.crs
        return str(crs) if crs is not None else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------


def set_encoding(
    xrdata: xr.Dataset, compress_method: str = "zlib"
) -> dict[str, dict]:
    """Build a NetCDF encoding dictionary with zlib compression.

    Preserves ``grid_mapping`` attributes where they exist so that the CRS
    information survives a round-trip through ``.to_netcdf()``.

    Parameters
    ----------
    xrdata : xr.Dataset
        Dataset whose variables will be encoded.
    compress_method : str
        Compression method.  Currently only ``"zlib"`` is supported by the
        netCDF4 backend.  Default: ``"zlib"``.

    Returns
    -------
    dict[str, dict]
        Encoding dict ready to pass to ``xr.Dataset.to_netcdf(encoding=...)``.
    """
    encoding: dict[str, dict] = {}
    for var in xrdata.data_vars:
        encoding[var] = {compress_method: True}
        if "grid_mapping" in xrdata[var].attrs:
            encoding[var]["grid_mapping"] = xrdata[var].attrs["grid_mapping"]
            # Remove from attrs to prevent xarray raising a duplicate-key error
            del xrdata[var].attrs["grid_mapping"]

    if "spatial_ref" in xrdata.variables:
        encoding["spatial_ref"] = {}

    return encoding


# ---------------------------------------------------------------------------
# CRS consistency check
# ---------------------------------------------------------------------------


def check_crs_in_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Ensure the dataset has a consistent ``spatial_ref`` + ``grid_mapping``.

    After ``xr.concat`` or ``rio.reproject`` the CRS metadata can become
    inconsistent.  This function re-writes the CRS from the ``spatial_ref``
    coordinate (if present) and sets the ``grid_mapping`` attribute on all
    data variables.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with consistent CRS metadata.
    """
    try:
        import rioxarray  # noqa: F401

        if ds.rio.crs is not None:
            ds = ds.rio.write_crs(ds.rio.crs, inplace=True)
            for var in ds.data_vars:
                if var != "spatial_ref":
                    ds[var].attrs["grid_mapping"] = "spatial_ref"
    except Exception as exc:  # noqa: BLE001
        logger.warning("check_crs_in_dataset: could not fix CRS — %s", exc)
    return ds


# ---------------------------------------------------------------------------
# Clipping / masking functions
# ---------------------------------------------------------------------------


def clip_to_bbox(
    ds: xr.Dataset,
    xyxy: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
) -> xr.Dataset:
    """Clip a dataset to a bounding box using rioxarray.

    This is a lazy operation — it only selects the relevant chunks.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (may be lazy / Dask-backed).
    xyxy : tuple[float, float, float, float]
        ``(xmin, ymin, xmax, ymax)`` bounding box in the dataset's CRS.
    crs : str
        CRS of the bounding box coordinates.  Default: ``"EPSG:4326"``.

    Returns
    -------
    xr.Dataset
        Clipped dataset (still lazy).
    """
    import rioxarray  # noqa: F401

    x1, y1, x2, y2 = xyxy
    src_crs = _get_dataset_crs_string(ds)
    if src_crs is None:
        ds = ds.rio.write_crs(crs)
    return ds.rio.clip_box(minx=x1, miny=y1, maxx=x2, maxy=y2)


def reproject_dataset(
    ds: xr.Dataset,
    target_crs: str,
    resampling: str = "nearest",
) -> xr.Dataset:
    """Reproject a lazy dataset to a new CRS using rioxarray.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    target_crs : str
        Target CRS string (e.g. ``"EPSG:4326"``).
    resampling : str
        Rasterio resampling method.  Default: ``"nearest"``.

    Returns
    -------
    xr.Dataset
        Reprojected dataset.
    """
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling as RioResampling

    method = (
        RioResampling.nearest if resampling == "nearest" else RioResampling.bilinear
    )
    logger.debug("Reprojecting to %s (method=%s)", target_crs, resampling)
    return ds.rio.reproject(target_crs, resampling=method)


def mask_with_geometry(
    ds: xr.Dataset,
    geometry: gpd.GeoDataFrame,
    clip: bool = True,
    all_touched: bool = True,
    use_rio: bool = False,
) -> xr.Dataset:
    """Mask a dataset using a GeoDataFrame geometry.

    Two backends are available:

    * **rasterio** (``use_rio=True``) — uses ``rioxarray.rio.clip``.  More
      accurate at polygon edges but slower for large temporal dimensions.
    * **geopandas mask** (``use_rio=False``) — builds a 2-D boolean mask with
      ``rasterio.features.geometry_mask`` and broadcasts it across all time /
      depth dimensions without loading the data.  **Preferred** for 3-D cubes.

    Parameters
    ----------
    ds : xr.Dataset
        Lazy input dataset.
    geometry : gpd.GeoDataFrame
        Masking geometry (any CRS; will be reprojected to match the dataset).
    clip : bool
        If ``True``, clip the bbox after masking.  Default: ``True``.
    all_touched : bool
        Include pixels touched at the boundary.  Default: ``True``.
    use_rio : bool
        Use rioxarray ``rio.clip`` instead of the geopandas mask.
        Default: ``False``.

    Returns
    -------
    xr.Dataset
        Masked (and optionally clipped) dataset.
    """
    import rasterio.features
    import rioxarray  # noqa: F401
    from shapely.geometry import mapping

    src_crs = _get_dataset_crs_string(ds)

    if use_rio:
        if src_crs is not None:
            geom = geometry.to_crs(src_crs)
        else:
            geom = geometry
        ds = ds.rio.write_crs(src_crs or "EPSG:4326")
        try:
            return ds.rio.clip(
                geom.geometry.apply(mapping),
                geom.crs,
                drop=clip,
                all_touched=all_touched,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("rio.clip failed: %s", exc)
            raise

    # Geopandas mask path (faster for 3-D cubes)
    try:
        src_transform = ds.rio.transform()
    except Exception:  # noqa: BLE001
        src_transform = ds.attrs.get("transform")

    # Determine spatial dim names
    dims = list(ds.sizes.keys())
    y_dim = next((d for d in dims if d in {"y", "lat", "latitude"}), dims[-2])
    x_dim = next((d for d in dims if d in {"x", "lon", "longitude"}), dims[-1])

    # GeoDataFrames iterate over column names, not geometries — extract explicitly.
    if isinstance(geometry, gpd.GeoDataFrame):
        geom_shapes = [mapping(g) for g in geometry.geometry]
    elif hasattr(geometry, "__iter__"):
        geom_shapes = [mapping(g) if hasattr(g, "__geo_interface__") else g for g in geometry]
    else:
        geom_shapes = [mapping(geometry)]

    shape_mask = rasterio.features.geometry_mask(
        geom_shapes,
        out_shape=(len(ds[y_dim]), len(ds[x_dim])),
        transform=src_transform,
        all_touched=all_touched,
        invert=True,
    )
    mask_da = xr.DataArray(shape_mask, dims=(y_dim, x_dim))
    masked = ds.where(mask_da)

    if clip:
        x1, y1, x2, y2 = geometry.total_bounds
        masked = clip_to_bbox(masked, (x1, y1, x2, y2))

    return masked


# ---------------------------------------------------------------------------
# High-level ROI extractor
# ---------------------------------------------------------------------------


def get_roi_data(
    ds: xr.Dataset,
    feature_geometry: gpd.GeoDataFrame,
    xyxy: tuple[float, float, float, float] | None = None,
    clip: bool = True,
    all_touched: bool = True,
    use_rio: bool = False,
    target_crs: str | None = None,
) -> xr.Dataset:
    """Extract a Region-Of-Interest (ROI) from a lazy dataset.

    Workflow
    --------
    1. If ``xyxy`` is provided, first clip the dataset to the bounding box
       (cheap — touches only the relevant Dask chunks).
    2. Reproject the dataset to ``target_crs`` if specified and the source
       CRS differs.
    3. Apply the geometry mask using :func:`mask_with_geometry`.

    CRS resolution is handled by ``@lru_cache``-backed :func:`_resolve_crs`
    so repeated calls in a pixel loop pay zero pyproj parsing overhead.

    Parameters
    ----------
    ds : xr.Dataset
        Lazy input dataset (climate or soil cube).
    feature_geometry : gpd.GeoDataFrame
        GeoDataFrame with the ROI geometry.  Can contain one or more polygons.
    xyxy : tuple | None
        Optional pre-clip bounding box ``(xmin, ymin, xmax, ymax)``.
        If ``None``, derived from ``feature_geometry.total_bounds``.
    clip : bool
        Clip the dataset to the geometry bbox after masking.  Default: True.
    all_touched : bool
        Touch-boundary pixel inclusion.  Default: True.
    use_rio : bool
        Use ``rioxarray.rio.clip`` backend.  Default: False.
    target_crs : str | None
        Reproject to this CRS before masking.  ``None`` → no reprojection.

    Returns
    -------
    xr.Dataset
        Masked, clipped, lazy dataset for the ROI.
    """
    import rioxarray  # noqa: F401

    # ------------------------------------------------------------------
    # Step 1: Optional coarse spatial clip (fast — no compute)
    # ------------------------------------------------------------------
    if xyxy is None:
        x1, y1, x2, y2 = feature_geometry.total_bounds
    else:
        x1, y1, x2, y2 = xyxy

    src_crs = _get_dataset_crs_string(ds)
    roi = clip_to_bbox(ds, (x1, y1, x2, y2), crs=src_crs or "EPSG:4326")

    # ------------------------------------------------------------------
    # Step 2: Optional reprojection (lazy via rioxarray)
    # ------------------------------------------------------------------
    if target_crs is not None and src_crs is not None:
        src_crs_obj = _resolve_crs(src_crs)
        tgt_crs_obj = _resolve_crs(target_crs)
        if src_crs_obj != tgt_crs_obj:
            logger.debug("Reprojecting from %s to %s", src_crs, target_crs)
            roi = reproject_dataset(roi, target_crs)

    # ------------------------------------------------------------------
    # Step 3: Geometry mask (still lazy until caller calls .compute())
    # ------------------------------------------------------------------
    roi = mask_with_geometry(
        roi, feature_geometry, clip=clip, all_touched=all_touched, use_rio=use_rio
    )

    return roi


# ---------------------------------------------------------------------------
# Spatial resolution rescaling
# ---------------------------------------------------------------------------


def rescale_dataset(
    ds: xr.Dataset,
    scale_factor: int,
    method: str = "nearest",
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.Dataset:
    """Spatially rescale a dataset by a scale factor using xarray interpolation.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    scale_factor : int
        Upscale (> 1) or downscale (< 1) factor.
    method : str
        Interpolation method.  Default: ``"nearest"``.
    x_dim : str
        Name of the x dimension.  Default: ``"x"``.
    y_dim : str
        Name of the y dimension.  Default: ``"y"``.

    Returns
    -------
    xr.Dataset
        Rescaled dataset.
    """
    import rioxarray  # noqa: F401

    if scale_factor == 1:
        return ds  # no-op

    old_x = ds[x_dim].values
    old_y = ds[y_dim].values

    n_x = int(len(old_x) * scale_factor)
    n_y = int(len(old_y) * scale_factor)

    new_x = np.linspace(old_x.min(), old_x.max(), n_x)
    new_y = np.linspace(old_y.min(), old_y.max(), n_y)

    rescaled = ds.interp({x_dim: new_x, y_dim: new_y}, method=method)
    logger.debug("Rescaled dataset: (%d×%d) → (%d×%d)", len(old_x), len(old_y), n_x, n_y)
    return rescaled


# ---------------------------------------------------------------------------
# Boundary helper
# ---------------------------------------------------------------------------


def get_boundaries_from_shapefile(
    path: str,
    crs: str | None = None,
    round_numbers: bool = False,
) -> tuple[float, float, float, float]:
    """Read a shapefile and return its total bounding box.

    Parameters
    ----------
    path : str
        Path to a shapefile or GeoPackage.
    crs : str | None
        Reproject to this CRS before computing bounds.  ``None`` → keep native.
    round_numbers : bool
        Round bounds outward to the nearest integer.  Default: ``False``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(xmin, ymin, xmax, ymax)``
    """
    features = gpd.read_file(path)
    if crs:
        features = features.to_crs(crs)

    x1, y1, x2, y2 = features.total_bounds
    if round_numbers:
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.ceil(x2)
        y2 = math.ceil(y2)

    return float(x1), float(y1), float(x2), float(y2)
