"""
ag_cube_cm.transform.soil_cube
================================

Transform Layer — Soil Datacube Builder.

**Rule:** This module transforms downloaded raw SoilGrids GeoTIFF files into
a single multi-depth NetCDF datacube suitable for DSSAT, CAF2021, and SIMPLE
model ingestion.  It does NOT download any data.

Key classes and functions
-------------------------
* :class:`SoilDataCubeBuilder`  — reads per-depth folders of GeoTIFF files,
  co-registers them to a common spatial reference, and concatenates them
  along a ``depth`` dimension.
* :func:`create_depth_dimension` — low-level stacking helper
* :func:`calculate_rgf`  — Root Growth Factor calculation for soil layers.
* :func:`find_soil_textural_class_in_nparray` — vectorised USDA texture
  classification on 2-D numpy arrays.
* :func:`get_layer_texture` — add a ``texture`` layer to a soil dataset.

Soil profile depths
-------------------
SoilGrids uses the ISRIC standard ``["0-5", "5-15", "15-30", "30-60",
"60-100", "100-200"]`` depth intervals.  The cube dimension ``depth``
is stored as the string labels (e.g. ``"0-5"``) sorted by the upper bound.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import xarray as xr
import geopandas as gpd

from ag_cube_cm.ingestion.files_manager import SoilFolderManager  # legacy helper
from ag_cube_cm.ingestion.utils import resample_variables  # legacy helper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Texture classification
# ---------------------------------------------------------------------------

#: USDA NRCS soil textural class codes (0 = unknown).
TEXTURE_CLASSES: dict[int, str] = {
    0: "unknown",
    1: "sand",
    2: "loamy sand",
    3: "sandy loam",
    4: "loam",
    5: "silt loam",
    6: "silt",
    7: "sandy clay loam",
    8: "clay loam",
    9: "silty clay loam",
    10: "sandy clay",
    11: "silty clay",
    12: "clay",
}


def find_soil_textural_class_in_nparray(
    sand: np.ndarray, clay: np.ndarray
) -> np.ndarray:
    """Vectorised USDA soil texture classification on 2-D NumPy arrays.

    Parameters
    ----------
    sand : np.ndarray
        Percentage sand content (0–100).
    clay : np.ndarray
        Percentage clay content (0–100).

    Returns
    -------
    np.ndarray
        Integer array of USDA texture class codes (see :data:`TEXTURE_CLASSES`).

    Raises
    ------
    TypeError
        If ``sand`` is not a NumPy array.
    """
    if not isinstance(sand, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(sand)}")

    silt = 100 - sand - clay
    silt[silt == 100] = 0

    cond1 = (sand >= 85) & ((silt + clay * 1.5) < 15)
    cond2 = (sand > 70) & (sand < 91) & ((silt + 1.5 * clay) >= 15) & ((silt + 2 * clay) < 30)
    cond3 = (
        ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2 * clay) >= 30))
        | ((clay < 7) & (silt < 50) & (sand > 43))
    )
    cond4 = (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52)
    cond5 = ((silt >= 50) & (clay >= 12) & (clay < 27)) | (
        (silt >= 50) & (silt < 80) & (clay < 12)
    )
    cond6 = (silt >= 80) & (clay < 12)
    cond7 = (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)
    cond8 = (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)
    cond9 = (clay >= 27) & (clay < 40) & (sand <= 20)
    cond10 = (clay >= 35) & (sand > 45)
    cond11 = (clay >= 40) & (silt >= 40)
    cond12 = (clay >= 40) & (sand <= 45) & (silt < 40)

    result = np.zeros(clay.shape, dtype=int)
    result[clay == 0] = -1

    for code, cond in enumerate(
        [cond1, cond2, cond3, cond4, cond5, cond6,
         cond7, cond8, cond9, cond10, cond11, cond12],
        start=1,
    ):
        result[(result == 0) & cond] = code

    result[result == -1] = 0
    return result


# ---------------------------------------------------------------------------
# Root Growth Factor
# ---------------------------------------------------------------------------


def calculate_rgf(depths: list[int]) -> list[float]:
    """Calculate Root Growth Factor (RGF) for soil layers.

    RGF ranges from 0.0 (no root growth) to 1.0 (unrestricted) and decays
    exponentially with depth below 15 cm.

    Parameters
    ----------
    depths : list[int]
        Layer bottom depths in cm.

    Returns
    -------
    list[float]
        RGF values, one per depth layer.

    Examples
    --------
    >>> calculate_rgf([5, 15, 30, 60])
    [1, 1, 0.834..., 0.549...]
    """
    arr = np.array(depths)
    if len(arr) > 1:
        centres: list[float] = [float(arr[0] / 2)] + (
            ((arr[1:] - arr[:-1]) / 2 + arr[:-1]).tolist()
        )
    else:
        centres = list(arr.astype(float))

    return [1.0 if c <= 15 else float(np.exp(-0.02 * c)) for c in centres]


# ---------------------------------------------------------------------------
# Depth dimension stacking
# ---------------------------------------------------------------------------


def create_depth_dimension(
    xrdata_dict: dict[str, xr.Dataset],
    dim_name: str = "depth",
) -> xr.Dataset:
    """Concatenate a per-depth dict of datasets along a new depth dimension.

    The CRS from the first dataset is restored on the resulting datacube
    (``xr.concat`` strips CRS metadata in some xarray versions).

    Parameters
    ----------
    xrdata_dict : dict[str, xr.Dataset]
        Mapping of ``{depth_label: xr.Dataset}``.  Keys become coordinate
        values along the ``depth`` dimension.
    dim_name : str
        Name for the new depth coordinate.  Default: ``"depth"``.

    Returns
    -------
    xr.Dataset
        Multi-depth dataset with ``band_data`` variable removed if present.
    """
    first_ds = list(xrdata_dict.values())[0]
    reference_crs = None
    try:
        import rioxarray  # noqa: F401

        reference_crs = first_ds.rio.crs
    except Exception:  # noqa: BLE001
        pass

    slices: list[xr.Dataset] = []
    for label, ds in tqdm.tqdm(xrdata_dict.items(), desc="Stacking soil depths"):
        ds_exp = ds.expand_dims(dim=[dim_name])
        ds_exp[dim_name] = [label]
        slices.append(ds_exp)

    cube = xr.concat(slices, dim=dim_name)

    # Restore CRS
    if reference_crs is not None:
        try:
            cube.rio.write_crs(reference_crs, inplace=True)
            for var in cube.data_vars:
                if var != "spatial_ref":
                    cube[var].attrs["grid_mapping"] = "spatial_ref"
        except Exception:  # noqa: BLE001
            pass

    # Remove legacy band_data variable if xarray added it
    if "band_data" in cube.data_vars:
        cube = cube.drop_vars("band_data")

    return cube


# ---------------------------------------------------------------------------
# Texture layer helper
# ---------------------------------------------------------------------------


def get_layer_texture(
    soil_layer: xr.Dataset,
    texture_name: str = "texture",
) -> xr.Dataset:
    """Add a USDA soil textural class layer to a soil dataset.

    Parameters
    ----------
    soil_layer : xr.Dataset
        Dataset containing ``sand`` and ``clay`` variables (% by weight).
    texture_name : str
        Name of the new texture variable.  Default: ``"texture"``.

    Returns
    -------
    xr.Dataset
        Original dataset with an additional ``texture`` variable.
    """
    from ag_cube_cm.ingestion.gis_functions import add_2dlayer_toxarrayr  # legacy

    # Scale if stored as g/kg (SoilGrids raw values > 100)
    sand = soil_layer["sand"].values
    clay = soil_layer["clay"].values
    if np.nanmax(sand) > 100:
        sand = sand * 0.1
        clay = clay * 0.1

    texture_map = find_soil_textural_class_in_nparray(sand, clay).astype(float)
    texture_map[texture_map == 0] = np.nan

    return add_2dlayer_toxarrayr(texture_map, soil_layer.copy(), variable_name=texture_name)


# ---------------------------------------------------------------------------
# Main soil cube builder
# ---------------------------------------------------------------------------


class SoilDataCubeBuilder:
    """Build a multi-depth soil NetCDF datacube from raw SoilGrids GeoTIFFs.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the downloaded SoilGrids GeoTIFFs
        (one file per variable+depth combination).
    variables : list[str]
        Soil variable names to include (e.g. ``["clay", "sand", "wv1500"]``).
    extent : list[float] | None
        Optional spatial clipping extent ``[xmin, ymin, xmax, ymax]``.
    reference_variable : str
        Variable used as the spatial resolution reference for co-registration.
        Default: ``"wv1500"``.
    crs : str
        Native CRS of the SoilGrids files.  Default: ``"ESRI:54052"``
        (Interrupted Goode Homolosine).
    target_crs : str | None
        Reproject the final cube to this CRS.  ``None`` → keep native.
        Default: ``"EPSG:4326"``.

    Examples
    --------
    >>> builder = SoilDataCubeBuilder(
    ...     data_folder="data/raw/soil_mwi",
    ...     variables=["clay", "sand", "bdod", "wv1500"],
    ...     target_crs="EPSG:4326",
    ... )
    >>> cube = builder.build()
    >>> cube.to_netcdf("data/soil_mwi.nc", encoding=builder.encoding(cube))
    """

    def __init__(
        self,
        data_folder: str,
        variables: list[str],
        extent: list[float] | None = None,
        reference_variable: str = "wv1500",
        crs: str = "ESRI:54052",
        target_crs: str | None = "EPSG:4326",
    ) -> None:
        self.data_folder = data_folder
        self.variables = variables
        self._extent = extent
        self.reference_variable = reference_variable
        self.crs = crs
        self.target_crs = target_crs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, verbose: bool = True) -> xr.Dataset:
        """Build the multi-depth soil datacube.

        Returns
        -------
        xr.Dataset
            Multi-depth dataset with dimension ``depth``.
        """
        folder_manager = SoilFolderManager(self.data_folder, self.variables)
        query_paths = folder_manager.get_all_paths(by="depth")

        xr_by_depth: dict[str, xr.Dataset] = {}
        for depth_label, var_paths in tqdm.tqdm(
            query_paths.items(), desc="Building soil cube", disable=not verbose
        ):
            xr_by_depth[depth_label] = self._stack_depth_layer(
                var_paths, verbose=verbose
            )

        cube = create_depth_dimension(xr_by_depth, dim_name="depth")

        if self.target_crs is not None:
            try:
                import rioxarray  # noqa: F401

                cube.rio.write_crs(self.target_crs, inplace=True)
                logger.info("CRS written: %s", self.target_crs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write target CRS: %s", exc)

        return cube

    def build_and_save(
        self,
        output_path: str,
        filename: str | None = None,
        verbose: bool = True,
    ) -> str:
        """Build the soil datacube and save it to a NetCDF file.

        Parameters
        ----------
        output_path : str
            Directory where the output file will be saved.
        filename : str | None
            Output file name.  If ``None``, derived from the data folder name.
        verbose : bool
            Show progress bars.  Default: ``True``.

        Returns
        -------
        str
            Path to the saved NetCDF file.
        """
        import rioxarray  # noqa: F401
        from ag_cube_cm.spatial.raster_ops import check_crs_in_dataset, set_encoding

        cube = self.build(verbose=verbose)

        # Ensure CRS is written to dataset attrs and spatial_ref variable
        if self.target_crs:
            try:
                cube = cube.rio.write_crs(self.target_crs, grid_mapping_name="spatial_ref")
                cube.attrs["crs"] = self.target_crs
            except Exception:  # noqa: BLE001
                pass

        cube = check_crs_in_dataset(cube)

        if filename is None:
            folder_name = Path(self.data_folder).name
            filename = f"soil_{folder_name}.nc"

        out_file = os.path.join(output_path, filename)
        encoding = set_encoding(cube)
        cube.to_netcdf(out_file, encoding=encoding, engine="netcdf4")
        logger.info("Soil datacube saved → %s", out_file)
        return out_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_depth_layer(
        self,
        var_paths: dict[str, str],
        verbose: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Load and co-register all variables for a single depth layer.

        Uses rioxarray directly so the native CRS is explicitly assigned
        (SoilGrids TIFs often have no embedded CRS metadata) before
        reprojection to ``target_crs`` and grid-matching to the reference.

        Parameters
        ----------
        var_paths : dict[str, str]
            ``{variable: file_path}`` for the current depth.

        Returns
        -------
        xr.Dataset
            Co-registered multi-variable dataset for this depth, in
            ``target_crs`` (default EPSG:4326).
        """
        import rioxarray  # noqa: F401

        data_arrays: dict[str, xr.DataArray] = {}
        ref_da: xr.DataArray | None = None

        for var, fp in var_paths.items():
            try:
                da = rioxarray.open_rasterio(fp, masked=True)
                if "band" in da.dims and da.sizes["band"] == 1:
                    da = da.squeeze("band", drop=True)

                # Assign CRS when the TIF has no embedded CRS metadata.
                # Two sources exist with different projections:
                #   *_30s.tif  — SoilGrids WCS → Homolosine (coordinates in metres, >1000)
                #   *_mean.tif — Google Storage → EPSG:4326 (coordinates in degrees, <180)
                if da.rio.crs is None:
                    x_mag = float(abs(da.x.values[0])) if da.x.size > 0 else 0.0
                    native_crs = self.crs if x_mag > 1000 else "EPSG:4326"
                    da = da.rio.write_crs(native_crs)

                # Reproject to target CRS
                if self.target_crs:
                    current = da.rio.crs
                    from pyproj import CRS as _CRS
                    if current and _CRS(str(current)) != _CRS(self.target_crs):
                        da = da.rio.reproject(self.target_crs)

                da.name = var
                data_arrays[var] = da

                if var == self.reference_variable:
                    ref_da = da

            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read %s (%s): %s", var, fp, exc)

        if not data_arrays:
            raise ValueError(
                f"No soil variables could be loaded for this depth layer. "
                f"Files attempted: {list(var_paths.values())}"
            )

        # Fall back to the first successfully loaded variable as reference
        if ref_da is None:
            ref_da = next(iter(data_arrays.values()))
            logger.warning(
                "Reference variable '%s' not loaded; using '%s' instead.",
                self.reference_variable, ref_da.name,
            )

        # Align all variables to the reference grid
        merged: dict[str, xr.DataArray] = {}
        for var, da in data_arrays.items():
            try:
                merged[var] = ref_da if var == ref_da.name else da.rio.reproject_match(ref_da)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not resample %s to reference grid: %s", var, exc)

        ds = xr.Dataset(merged)
        if self.target_crs:
            try:
                ds.rio.write_crs(self.target_crs, inplace=True)
            except Exception:  # noqa: BLE001
                pass
        return ds

    @staticmethod
    def encoding(cube: xr.Dataset, compress_method: str = "zlib") -> dict[str, dict]:
        """Generate a zlib encoding dict for the soil datacube.

        Parameters
        ----------
        cube : xr.Dataset
            Soil datacube to encode.
        compress_method : str
            Compression method.  Default: ``"zlib"``.

        Returns
        -------
        dict[str, dict]
            Encoding dict for ``to_netcdf(encoding=...)``.
        """
        from ag_cube_cm.spatial.raster_ops import set_encoding

        return set_encoding(cube, compress_method=compress_method)
