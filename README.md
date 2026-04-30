# ag-cube-cm

**ag-cube-cm** is a memory-efficient Python package for spatial NetCDF datacube processing and process-based crop model orchestration. It lets agro-climatologists and spatial data scientists run pixel-level agricultural simulations at scale — either programmatically or through natural language via an AI assistant.

## Features

- **Spatial Data Engineering**: Downloads and processes climate data (AgERA5 via Copernicus CDS, CHIRPS) and soil properties (SoilGrids WCS) using `xarray` and `dask`.
- **Strict Configuration**: `Pydantic v2` validates all YAML inputs (`SimulationConfig`) — crop selection, agronomic management, spatial domain.
- **Process-Based Crop Models**: Registry-based factory pattern (`CropModel` ABC). Supported models:
  - **DSSAT** — industry-standard crop model, pixel-isolated working directories, C-mode execution.
  - **Banana_N** — pure-Python banana growth model, runs entirely in memory.
  - **CAF2021** — agroforestry / coffee-shade model.
  - **SIMPLE** — generic crop model for development and yield.
- **Parallel Orchestrator**: `ThreadPool`/`ProcessPool` across all `(pixel × planting-window)` combinations.
- **Universal Reporter**: Exports simulation outputs to NetCDF or Parquet.
- **MCP Server**: Exposes all operations as tools that an AI assistant can call via natural language.

---

## Installation

```bash
# Core package
pip install git+https://github.com/anaguilarar/ag-cube-cm.git

# With DSSAT / crop model support
pip install "ag-cube-cm[models] @ git+https://github.com/anaguilarar/ag-cube-cm.git"

# With data download support (requires CDS API key)
pip install "ag-cube-cm[download] @ git+https://github.com/anaguilarar/ag-cube-cm.git"

# With MCP server (for AI assistant integration)
pip install "ag-cube-cm[mcp] @ git+https://github.com/anaguilarar/ag-cube-cm.git"

# Everything
pip install "ag-cube-cm[all] @ git+https://github.com/anaguilarar/ag-cube-cm.git"
```

---

## Quick Start — Python API

### 1. Run a spatial DSSAT simulation

```python
from ag_cube_cm.config.loader import load_config
from ag_cube_cm.models.dssat.base import DSSATModel
import xarray as xr

cfg = load_config("options/my_config.yaml")
weather_ds = xr.open_dataset(cfg.SPATIAL_INFO.weather_path)
soil_ds    = xr.open_dataset(cfg.SPATIAL_INFO.soil_path)

# Single pixel
w_slice = weather_ds.sel(y=-13.5, x=34.0, method="nearest")
s_slice = soil_ds.sel(y=-13.5, x=34.0, method="nearest")

model = DSSATModel(cfg)
model.setup_working_directory("test_px")
model.prepare_inputs(w_slice, s_slice, elevation=0.0)
model.run_simulation()
print(model.collect_outputs())   # {"HWAM": 4215.0, ...}
model.cleanup_working_directory()
```

### 2. Full spatial run via script

```bash
python examples/spatial_dssat_run.py --config options/test_dssat.yaml
```

### 3. YAML config example

```yaml
GENERAL_INFO:
  country: 'Malawi'
  country_code: 'MWI'
  model: 'dssat'
  working_path: '/tmp/mlw_dssat'    # ← NO spaces in this path
  dssat_path: null                  # null = use bundled binary
  ncores: 8

SPATIAL_INFO:
  feature_name: 'shapeName'
  soil_path: 'data/soil_mwi.nc'
  weather_path: 'data/weather_mwi_2000_2019.nc'
  output_path: 'malawi_yield.nc'
  dem_path: null

CROP:
  name: 'Maize'
  cultivar: 'IB1072'

MANAGEMENT:
  planting_date: '2000-11-01'
  n_planting_windows: 6
  planting_window_days: 7
  fertilizer_schedule:
    - days_after_planting: 5
      n_kg_ha: 50.0
      p_kg_ha: 30.0
```

> **Important:** `working_path` must not contain spaces. DSSAT's `DSSATPRO.V48`
> config file is whitespace-delimited; a space in the path corrupts the model
> entry and causes a silent `rc=99` error.

---

## AI-Assisted Workflows (MCP + Skill)

The package ships an MCP server that exposes all operations as tools.
Once registered, an AI assistant (Claude + the `spatial-crop-modeler` skill)
can drive the full workflow — data download, config generation, simulation —
from a single natural-language request.

### Step 1 — Install MCP support

```bash
pip install "ag-cube-cm[mcp] @ git+https://github.com/anaguilarar/ag-cube-cm.git"
```

### Step 2 — Register the server with Claude Code

Create `.claude/mcp_config.json` in your project root (or in
`~/.claude/mcp_config.json` for global access):

```json
{
  "mcpServers": {
    "ag-cube-cm": {
      "command": "python",
      "args": ["-m", "ag_cube_cm.mcp_server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      },
      "description": "Download AgERA5/CHIRPS/SoilGrids data and run DSSAT crop simulations."
    }
  }
}
```

Or register it from the Claude Code CLI:

```bash
claude mcp add ag-cube-cm python -- -m ag_cube_cm.mcp_server
```

### Step 3 — Load the skill and start talking

Open Claude Code in your project directory and invoke the skill:

```
/spatial-crop-modeler
```

Then describe what you want in plain language. The assistant asks clarifying
questions, then calls the MCP tools in the correct order automatically.

### Example conversation

```
You:     I want to simulate maize yield in Malawi for 2000-2019,
         planting in November with 6 windows, no fertilizer.

Claude:  What working path should DSSAT use? (must have no spaces)
         Do you already have weather and soil files, or should I download them?

You:     Download everything. Save runs to /tmp/mlw_dssat.

Claude:  [calls download_weather]  → weather_mwi.nc downloaded (7305 days)
         [calls download_soil]     → soil_mwi.nc downloaded (5 depths)
         [calls generate_config]   → malawi.yaml created
         [calls run_simulation]    → 16560 pixels × 6 windows running...

         Results saved to malawi_yield.nc
         ok=9840  skip=6720  failed=0
         Mean HWAM: 3 847 kg/ha

         Per-window summary:
           W00 (2000-11-01): mean=3 612  min=890   max=6 240 kg/ha
           W01 (2000-11-08): mean=3 920  ...
```

### Available MCP tools

| Tool | What it does |
|------|-------------|
| `list_supported_crops` | Lists all supported crops and example cultivar IDs |
| `download_weather` | Downloads AgERA5 or CHIRPS weather → single NetCDF datacube |
| `download_soil` | Downloads SoilGrids soil layers → GeoTIFF / NetCDF |
| `generate_config` | Generates and saves a simulation YAML config |
| `run_simulation` | Runs the crop model on all pixels × planting windows |

### Start the server manually

```bash
python -m ag_cube_cm.mcp_server
```

---

## Package Architecture

```
src/ag_cube_cm/
├── config/          Pydantic v2 SimulationConfig — YAML validation
├── ingestion/       AgERA5Downloader, CHIRPSDownloader, SoilGridsDownloader
├── models/
│   ├── base.py      CropModel ABC
│   ├── dssat/       DSSATModel — Fortran file writers, subprocess runner
│   ├── banana_n/    Pure-Python banana model
│   └── factory.py   @register_model decorator
├── spatial/         SpatialCM orchestrator, SpatialReporter
└── mcp_server.py    FastMCP server (AI assistant integration)
```

---

## Supported Crops (DSSAT)

| Crop | Code | | Crop | Code |
|------|----- |-|------|------|
| Maize | MZ | | Bean | BN |
| Wheat | WH | | Cassava | CS |
| Rice | RI | | Potato | PT |
| Sorghum | SG | | Sugarcane | SC |
| Millet | ML | | Sunflower | SU |
| Soybean | SB | | Canola | CN |

---

## Requirements

- Python ≥ 3.10
- DSSAT 4.8 binary — bundled in `static/bin/` (Linux + Windows)
- CDS API key — required only for AgERA5 downloads (`~/.cdsapirc`)

---

## License

MIT
