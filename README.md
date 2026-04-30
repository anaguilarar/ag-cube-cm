# ag-cube-cm

**ag-cube-cm** is a robust, memory-efficient Python package designed for spatial NetCDF datacube processing and process-based crop model orchestration. It allows agro-climatologists and spatial data scientists to execute pixel-level agricultural simulations at scale.

## Features

- **Spatial Data Engineering**: Ingests and processes historic climate data (AgEra5 via Copernicus CDS, CHIRPS) and soil properties (SoilGrids via WCS) using `xarray` and `dask` for memory-efficient datacube operations.
- **Strict Configuration**: Uses `Pydantic v2` to strictly validate YAML user inputs (`SimulationConfig`) for spatial domains, crop selections, and agronomic management.
- **Process-Based Crop Models (PBCMs)**: Easily pluggable architecture using a registry-based factory pattern (`CropModel` Abstract Base Class). Currently supported models:
  - **DSSAT**: Industry standard model. Runs use strict UUID-isolated temporary directories to prevent I/O race conditions during parallel processing.
  - **Banana_N**: A pure-Python, object-oriented crop model simulating banana growth, phenology, biomass accumulation, and residue-based nitrogen mineralization (runs entirely in memory, bypassing the GIL).
  - **CAF2021**: Agroforestry model designed to simulate coffee growth and its interaction with shade trees.
  - **SIMPLE**: Generic crop model easily modified to simulate development, growth, and yield.
- **SpatialCM Orchestrator**: High-performance concurrency using `ThreadPool`/`ProcessPool` for massive spatial simulations.
- **Universal Reporter**: Exports processed simulation outputs directly to Parquet or NetCDF formats.

## Architecture Highlights

* **`config/`**: YAML validation engine enforcing strict agronomic practices (e.g., NPK fertilizer schedules, Days After Planting).
* **`ingestion/` & `transform/`**: Downloads raw NetCDFs and stacks them into lazy data cubes.
* **`spatial/`**: Contains the `spatial_cm.py` orchestrator and the `reporter.py` reporting engine.
* **`models/`**: Houses all implemented PBCMs.

## Installation

```bash
git clone https://github.com/anaguilarar/ag-cube-cm.git
cd ag-cube-cm
pip install .
```

## Example Configuration (YAML)

To run a simulation, you must define a YAML configuration file. Here is an example:

```yaml
GENERAL_INFO: 
  country: 'Malawi'
  country_code: 'MWI'
  working_path: 'runs' 
  ncores: 10
  model: 'banana_n'

SPATIAL_INFO:
  aggregate_by: pixel
  soil_path: data/soil_mwi.nc
  weather_path: data/weather_mwi_2000_2019.nc
  scale_factor: 1

CROP:
  name: Banana
  cultivar: 'Williams'

MANAGEMENT:
  planting_date: '2001-03-01'
  plantingWindow: 20
  fertilizer_schedule: 
    days_after_planting: [30, 60]
    npk: [[50, 0, 0], [50, 0, 0]]
```

*(Note: For spatial DSSAT runs, it is highly recommended to use `keep_intermediate_files: false` in your configuration to prevent file system inode exhaustion.)*

## License

MIT
