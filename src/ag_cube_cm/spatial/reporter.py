"""
ag_cube_cm.spatial.reporter
===========================

Universal Spatial Reporter for converting simulated outputs back into
spatial structures (tabular or raster).
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

class SpatialReporter:
    """
    Handles the aggregation and exportation of simulation results into 
    tabular formats (Parquet, CSV) and raster formats (NetCDF, GeoTIFF).
    """

    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs

    def _reconstruct_grid(self, results_list: List[Tuple[float, float, Dict[str, Any]]], reference_mask: xr.DataArray = None) -> xr.Dataset:
        """
        Reconstructs an Xarray Dataset from a list of tuple results.

        Parameters
        ----------
        results_list : List[Tuple[float, float, Dict[str, Any]]]
            List of (lat/y, lon/x, result_dict) tuples.
        reference_mask : xr.DataArray, optional
            If provided, the reconstructed dataset is reindexed to strictly match this spatial grid.

        Returns
        -------
        xr.Dataset
            The reconstructed datacube.
        """
        if not results_list:
            logger.warning("Empty results_list provided. Returning empty Dataset.")
            return xr.Dataset()

        # Extract into a list of dicts for pandas
        records = []
        for y, x, res_dict in results_list:
            if not res_dict:
                continue
            record = {'y': y, 'x': x}
            record.update(res_dict)
            records.append(record)

        if not records:
            return xr.Dataset()

        df = pd.DataFrame(records)
        
        index_cols = ['y', 'x']
        if 'PDAT' in df.columns:
            try:
                df['PDAT'] = pd.to_datetime(df['PDAT'])
                index_cols = ['PDAT', 'y', 'x']
            except Exception:
                pass
            
        df = df.set_index(index_cols)
        
        ds = xr.Dataset.from_dataframe(df)
        ds.rio.write_crs(self.crs, inplace=True)

        if reference_mask is not None:
            ds = ds.reindex(x=reference_mask.x, y=reference_mask.y, method='nearest', tolerance=1e-5)

        return ds

    def export_to_tabular(self, dataset: xr.Dataset, filepath: Union[str, Path], format: str = "parquet") -> None:
        """
        Exports the Xarray Dataset into a tabular format, dropping NaNs to save space.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to export.
        filepath : Union[str, Path]
            Destination path.
        format : str
            'parquet' or 'csv'.
        """
        logger.info(f"Exporting to {format.upper()} at {filepath}")
        df = dataset.to_dataframe().reset_index()
        
        # Drop spatial NaNs (pixels outside ROI or missing data)
        data_cols = [c for c in df.columns if c not in ['x', 'y', 'PDAT']]
        if data_cols:
            df = df.dropna(subset=data_cols, how='all')

        if format.lower() == "parquet":
            df.to_parquet(filepath, index=False)
        elif format.lower() == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported tabular format: {format}")

    def export_to_raster(self, dataset: xr.Dataset, filepath: Union[str, Path], format: str = "netcdf") -> None:
        """
        Exports the Xarray Dataset directly to a spatial raster file.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to export.
        filepath : Union[str, Path]
            Destination path.
        format : str
            'netcdf' or 'tif'.
        """
        logger.info(f"Exporting to {format.upper()} at {filepath}")
        
        if not dataset.rio.crs:
            dataset.rio.write_crs(self.crs, inplace=True)
            
        if 'x' in dataset.coords and 'y' in dataset.coords:
            dataset = dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")

        if format.lower() == "netcdf":
            dataset.to_netcdf(filepath)
        elif format.lower() in ["tif", "tiff"]:
            try:
                dataset.rio.to_raster(filepath)
            except Exception as e:
                logger.error(f"Failed to export full dataset to TIF. Reason: {e}. Attempting first variable only.")
                first_var = list(dataset.data_vars)[0]
                dataset[first_var].rio.to_raster(filepath)
        else:
            raise ValueError(f"Unsupported raster format: {format}")
