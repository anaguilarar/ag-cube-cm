# Project Conventions for WeatherSoilDataProcessor

This project involves spatial data, agronomic models (DSSAT, CAF), and complex configurations. Use the following conventions when documenting core objects:

## Common Types

- **`OmegaConf`**: Use `OmegaConf` or `DictConfig` for configuration objects. Mention which keys are expected (e.g., `config.GENERAL.ncores`).
- **Spatial Extents**: Always document as `[xmin, ymin, xmax, ymax]` and specify units (usually decimal degrees).
- **DSSAT Files**: When functions return or write DSSAT-compatible files (e.g., `.WTH`, `.SOL`), mention the file extension and the DSSAT version (v4.8).
- **`pd.DataFrame`**: Specify the required columns if the function expects a certain schema (e.g., "Must contain 'date', 'tmax', 'tmin' columns").

## Working Directory
Functions that write temporary files (like `SpatialCAF`) should document where these files are stored (`self._tmp_path`) and whether they are automatically cleaned up.

## Multiprocessing
If a function supports `ncores`, clarify if it uses `multiprocessing` or `joblib` and how it handles resource distribution.
