"""ag_cube_cm.spatial — Lazy spatial data access and raster operations."""
from .data import SpatialData, _open_dataset
from .raster_ops import (
    clip_to_bbox,
    reproject_dataset,
    mask_with_geometry,
    get_roi_data,
    set_encoding,
    check_crs_in_dataset,
    get_boundaries_from_shapefile,
)

__all__ = [
    "SpatialData",
    "_open_dataset",
    "clip_to_bbox",
    "reproject_dataset",
    "mask_with_geometry",
    "get_roi_data",
    "set_encoding",
    "check_crs_in_dataset",
    "get_boundaries_from_shapefile",
]
