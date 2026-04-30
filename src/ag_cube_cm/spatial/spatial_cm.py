"""
ag_cube_cm.spatial.spatial_cm
==============================

Spatial Orchestrator for Crop Models.
"""

import concurrent.futures
import logging
from typing import Optional, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ag_cube_cm.config.schemas import SimulationConfig
from ag_cube_cm.spatial.data import SpatialData
from ag_cube_cm.spatial.raster_ops import get_roi_data
from ag_cube_cm.spatial.reporter import SpatialReporter
from ag_cube_cm.models.factory import model_factory

logger = logging.getLogger(__name__)


class SpatialCM:
    """
    Main Spatial Orchestrator.
    
    Responsible for:
    1. Masking Phase 2 lazy datacubes to a specific Region of Interest (ROI).
    2. Finding valid pixels (where both soil and weather data exist).
    3. Spawning dynamic parallel workers (Threads for binary models, Processes for pure Python).
    4. Passing pixel-level data slices to the underlying `CropModel` ABC.
    5. Assembling the output yields into a new spatial datacube.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        
        # 1. Initialize the Lazy Spatial Datacubes
        logger.info("Initializing SpatialData manager...")
        weather_path = getattr(self.config.WEATHER, 'data_cube_path', None)
        soil_path = getattr(self.config.SOIL, 'data_cube_path', None)
        dem_path = getattr(self.config.GENERAL_INFO, 'dem_path', None)
        
        self.spatial_data = SpatialData(
            weather_path=weather_path,
            soil_path=soil_path,
            dem_path=dem_path,
            engine='netcdf4'
        )
        
        # 2. Instantiate the configured CropModel via the Factory
        self.model = model_factory(self.config)
        
        # 3. Execution configuration
        self.ncores = getattr(self.config.GENERAL_INFO, 'ncores', 4)
        self.keep_intermediate = getattr(self.config.GENERAL_INFO, 'keep_intermediate_files', False)
        
        # Determine executor type
        # Binary models (DSSAT, CAF) block on I/O (subprocess execution) -> ThreadPool is better
        # Pure Python models (SIMPLE) block on CPU -> ProcessPool is better
        model_name = self.config.GENERAL_INFO.model.lower()
        if model_name in ['dssat', 'caf']:
            self.executor_class = concurrent.futures.ThreadPoolExecutor
        else:
            self.executor_class = concurrent.futures.ProcessPoolExecutor
            
        # 4. Initialize Spatial Reporter
        target_crs = getattr(self.config.GENERAL_INFO, 'projected_crs', 'EPSG:4326')
        self.reporter = SpatialReporter(crs=target_crs)

    def run_spatial_simulation(self, roi: gpd.GeoDataFrame) -> xr.Dataset:
        """
        Run the simulation across all valid pixels in the ROI.
        
        Parameters
        ----------
        roi : gpd.GeoDataFrame
            The Region of Interest geometry.
            
        Returns
        -------
        xr.Dataset
            A datacube containing the aggregated model outputs (e.g., yields).
        """
        # 1. Extract and mask Data
        logger.info("Extracting data slices for ROI...")
        target_crs = getattr(self.config.GENERAL_INFO, 'projected_crs', 'EPSG:4326')
        
        # Mask weather and soil cubes lazily
        weather_m = get_roi_data(self.spatial_data.climate, roi, target_crs=target_crs)
        soil_m = get_roi_data(self.spatial_data.soil, roi, target_crs=target_crs)
        
        # 2. Identify Valid Pixels (Intersection of valid soil & weather)
        # We need the spatial structure to be identical for merging
        # This requires bringing the first time/depth slice into memory
        logger.info("Identifying valid spatial pixels...")
        w_first = weather_m.isel(date=0).compute()
        s_first = soil_m.isel(depth=0).compute()
        
        w_var = list(w_first.data_vars)[0]
        s_var = list(s_first.data_vars)[0]
        
        # Create a boolean mask where both datasets have non-null values
        valid_mask = w_first[w_var].notnull() & s_first[s_var].notnull()
        
        x_coords, y_coords = np.meshgrid(valid_mask.x.values, valid_mask.y.values)
        valid_x = x_coords[valid_mask.values]
        valid_y = y_coords[valid_mask.values]
        
        pixels_to_run = list(zip(range(len(valid_x)), valid_x, valid_y))
        logger.info(f"Found {len(pixels_to_run)} valid pixels to simulate.")
        
        if not pixels_to_run:
            raise ValueError("No valid pixels found in the intersection of Soil and Weather data.")

        # 3. Parallel Execution
        results = []
        logger.info(f"Starting execution using {self.executor_class.__name__} with {self.ncores} workers.")
        
        with tqdm(total=len(pixels_to_run), desc="Simulating Pixels") as pbar:
            with self.executor_class(max_workers=self.ncores) as executor:
                # Submit tasks
                future_to_pixel = {
                    executor.submit(
                        self._process_pixel,
                        px_id,
                        x,
                        y,
                        weather_m,
                        soil_m
                    ): (px_id, x, y)
                    for px_id, x, y in pixels_to_run
                }
                
                # Gather results
                for future in concurrent.futures.as_completed(future_to_pixel):
                    px_id, x, y = future_to_pixel[future]
                    try:
                        res_tuple = future.result()
                        if res_tuple is not None:
                            results.append(res_tuple)
                    except Exception as exc:
                        logger.error(f"Pixel {px_id} generated an exception: {exc}")
                    pbar.update(1)

        # 4. Aggregate Results into Xarray via SpatialReporter
        logger.info("Aggregating results into spatial datacube...")
        if not results:
            logger.warning("No successful simulations. Returning empty dataset.")
            return xr.Dataset()
            
        yield_cube = self.reporter._reconstruct_grid(results, reference_mask=valid_mask)
        
        # 5. Export based on config
        out_format = getattr(self.config.GENERAL_INFO, 'output_format', 'netcdf').lower()
        out_path = getattr(self.config.GENERAL_INFO, 'output_path', 'output_yield')
        
        if out_format in ['parquet', 'csv']:
            self.reporter.export_to_tabular(yield_cube, f"{out_path}.{out_format}", format=out_format)
        else:
            self.reporter.export_to_raster(yield_cube, f"{out_path}.{'nc' if out_format == 'netcdf' else 'tif'}", format=out_format)
        
        return yield_cube

    def _process_pixel(self, pixel_id: int, x: float, y: float, 
                       weather_cube: xr.Dataset, soil_cube: xr.Dataset) -> Optional[Tuple[float, float, dict]]:
        """
        The core pipeline executed per pixel by the parallel workers.
        """
        # We instantiate a new model instance per thread/process to keep state isolated
        local_model = model_factory(self.config)
        
        # 1. Setup isolated directory
        local_model.setup_working_directory(pixel_id=str(pixel_id))
        
        try:
            # 2. Extract 1D slices for this exact location
            w_slice = weather_cube.sel(x=x, y=y, method='nearest').compute()
            s_slice = soil_cube.sel(x=x, y=y, method='nearest').compute()
            
            # 3. Prepare inputs
            local_model.prepare_inputs(w_slice, s_slice)
            
            # 4. Run Model
            local_model.run_simulation()
            
            # 5. Collect Outputs
            output_dict = local_model.collect_outputs()
            return (y, x, output_dict)
            
        except Exception as e:
            logger.error(f"Error processing pixel {pixel_id}: {e}")
            return None
            
        finally:
            # 6. Aggressive Cleanup
            if not self.keep_intermediate:
                local_model.cleanup_working_directory()



    def close(self):
        """Cleanly shutdown resources."""
        self.spatial_data.close()
