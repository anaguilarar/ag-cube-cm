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

### Step 1 — Authenticate AgERA5 (Required for Weather Downloads)

To access AgERA5 data, users must provide account credentials. This requires two key pieces of information:

- **Email**: The email address used to register the AgERA5 account.
- **API Code**: A unique code available in the profile settings after account creation.

The login must be done through: [https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview).

The following Python snippet is used to authenticate and access AgERA5 data by creating your credentials file:

```python
import os

YOURUSERAPICODE = 'YOURAPI'
YOUREMAIL = 'YOUREMAIL'

with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
    f.write("url: https://cds.climate.copernicus.eu/api\nkey: {}\nemail: {}".format(YOURUSERAPICODE, YOUREMAIL))
```

### Step 2 — Install MCP support

```bash
pip install "ag-cube-cm[mcp] @ git+https://github.com/anaguilarar/ag-cube-cm.git"
```

### Step 3 — Register the server with Claude Code

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

### Step 4 — Start Claude Code and load the command

To use the custom skill, you must first have the `.claude` folder available locally. You can do this by cloning the repository:

```bash
git clone https://github.com/anaguilarar/ag-cube-cm.git
cd ag-cube-cm
```
*(Alternatively, you can simply download the `.claude` folder into your working directory).*

Next, start Claude Code in that directory:

```bash
claude
```

Once Claude Code is running, invoke the bundled slash command:

```
/spatial-crop-modeler
```

Then describe what you want in plain language. The assistant asks clarifying
questions, then calls the MCP tools in the correct order automatically.

### Example conversation

```
You:     I want to simulate maize yield in Mwanza district, Malawi for 2010-2012,
         planting on 2010-11-01 with 5 windows, no fertilizer. Use 6 cores.

Claude:  What working path should DSSAT use? (must have no spaces)
         Do you already have weather and soil files, or should I download them?

You:     Download everything. Save runs to D:/tmp_dssat.

Claude:  [list_admin_units(country_code="MWI")]  → verifying 'Mwanza' is a valid district

         [download_weather(country_code="MWI", year_start=2010, year_end=2012,
                           feature="Mwanza", ncores=6)]
         → downloads Mwanza extent only, builds weather_2010_2012.nc

         [download_soil(country_code="MWI", feature="Mwanza")]
         → builds soil_mwi.nc

         [generate_config(..., feature="Mwanza", ncores=6,
                          save_to="D:/tmp_dssat/mwanza.yaml")]
         → mwanza.yaml created

         [run_simulation(config_path="D:/tmp_dssat/mwanza.yaml", feature="Mwanza")]
         → 85 pixels × 5 windows simulated

         Results saved to mwanza_yield.nc
         ok=425  skip=0  failed=0
         Mean HWAM: 4 120 kg/ha

         Per-window summary:
           W00 (2010-11-01): mean=4 050  min=1 200  max=6 100 kg/ha
           W01 (2010-11-08): mean=4 150  ...
```

### Available MCP tools

| Tool | Key parameters | What it does |
|------|---------------|-------------|
| `list_supported_crops` | — | Lists all supported crops and example cultivar IDs |
| `list_admin_units` | `country_code`, `adm_level` | Lists district / department names for a country — call before download to verify `feature` names |
| `download_weather` | `country_code`, `year_start`, `year_end`, `source` (`"agera5"` / `"chirps"`), `feature`, `adm_level`, `ncores` | Downloads AgERA5 + CHIRPS weather data clipped to the region extent and builds a merged multi-temporal NetCDF datacube. Returns `output_path` (the `.nc` file). |
| `download_soil` | `country_code`, `feature`, `adm_level`, `depths`, `variables` | Downloads SoilGrids layers clipped to the region extent and builds a multi-depth NetCDF datacube. Returns `output_path` (the `.nc` file). |
| `generate_config` | `country`, `country_code`, `model`, `weather_path`, `soil_path`, `crop`, `cultivar`, `planting_date`, `feature`, `adm_level`, `ncores`, `save_to` | Generates and saves a simulation YAML config. Always pass `save_to` — without it the file is never written to disk. |
| `run_simulation` | `config_path`, `feature`, `adm_level`, `max_pixels` | Runs the crop model on all pixels × planting windows. `feature` clips the datacubes to a district boundary before building the pixel list. |

#### Output chaining

| After calling | Read field | Pass as |
|---------------|-----------|---------|
| `download_weather` | `output_path` | `weather_path` in `generate_config` |
| `download_soil` | `output_path` | `soil_path` in `generate_config` |
| `generate_config` | `save_path` | `config_path` in `run_simulation` |

### Sub-country simulations (district / department level)

Running a full country is expensive (thousands of pixels). Pass `feature` to **every** tool
that accepts it to restrict downloads, config, and simulation to one district:

```
You:     Simulate maize in Mwanza district, Malawi, 2010-2012,
         planting 2010-11-01, 4 windows, no fertilizer, 6 cores.

Claude:  [list_admin_units(country_code="MWI", adm_level=2)]
         → confirms "Mwanza" is valid

         [download_weather(country_code="MWI", year_start=2010, year_end=2012,
                           feature="Mwanza", adm_level=2, ncores=6)]
         → downloads only the Mwanza extent (8 km buffered, WGS84)
         → builds weather_2010_2012.nc  ← returned as output_path

         [download_soil(country_code="MWI", feature="Mwanza", adm_level=2)]
         → builds soil_mwi.nc  ← returned as output_path

         [generate_config(..., feature="Mwanza", adm_level=2, ncores=6,
                          save_to="D:/tmp/mwanza.yaml")]
         → mwanza.yaml created

         [run_simulation(config_path="D:/tmp/mwanza.yaml",
                         feature="Mwanza", adm_level=2)]
         → clips both datacubes to Mwanza polygon
         → 85 pixels × 4 windows running...
         → mean HWAM: 4 120 kg/ha
```

**How the spatial clipping works:** `feature` triggers a GeoBoundaries API lookup,
projects the polygon to the SoilGrids Homolosine CRS (ESRI:54052), adds an 8 km buffer
to cover edge pixels at ~5 km resolution, then reprojects to WGS84 (1-decimal precision).
No local shapefile is needed. This typically reduces download size and pixel count by 10–100×.

### Start the server manually

```bash
python -m ag_cube_cm.mcp_server
```

---

## Package Architecture

```
src/ag_cube_cm/
├── config/          Pydantic v2 SimulationConfig — YAML validation
├── ingestion/       AgERA5Downloader, CHIRPSDownloader, SoilGridsDownloader, get_admin_boundary
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
