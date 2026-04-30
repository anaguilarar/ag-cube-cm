"""
ag_cube_cm.config.loader
=========================

YAML → Pydantic configuration loader.

This module provides a single public function, :func:`load_config`, that:

1. Reads a YAML file from disk using **OmegaConf** (preserving the existing
   toolchain) or falls back to plain **PyYAML** if OmegaConf is unavailable.
2. Recursively resolves any ``${...}`` variable interpolations (OmegaConf
   feature).
3. Converts the resolved dict to a plain Python ``dict`` (no OmegaConf
   internals leak into Pydantic).
4. Validates and constructs the :class:`~ag_cube_cm.config.schemas.SimulationConfig`
   Pydantic model, which provides a clear ``ValidationError`` for any missing
   or malformed field.

Usage
-----
.. code-block:: python

    from ag_cube_cm.config.loader import load_config

    cfg = load_config("configs/maize_dssat_example.yaml")

    # Rich, typed access — no more cfg["GENERAL_INFO"]["model"]
    print(cfg.GENERAL_INFO.model)        # 'dssat'
    print(cfg.MANAGEMENT.total_n_kg_ha)  # 300.0
    print(cfg.MANAGEMENT.planting_date)  # datetime.date(1991, 3, 1)

Error handling
--------------
``load_config`` wraps ``pydantic.ValidationError`` in a more user-friendly
:class:`ConfigValidationError` that includes the config file path in the
message, making it easier to debug in long CI logs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schemas import SimulationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """
    Raised when the YAML configuration file fails Pydantic validation.

    Wraps :class:`pydantic.ValidationError` and prepends the config file
    path so error messages are immediately actionable.

    Attributes
    ----------
    path : str
        Absolute path to the config file that triggered the error.
    original : pydantic.ValidationError
        The underlying Pydantic validation error.
    """

    def __init__(self, path: str, original: ValidationError) -> None:
        self.path = path
        self.original = original
        error_count = original.error_count()
        errors_detail = original.errors(include_url=False)
        summary_lines = [
            f"Configuration file '{path}' has {error_count} validation "
            f"error{'s' if error_count != 1 else ''}:\n"
        ]
        for err in errors_detail:
            loc = " → ".join(str(p) for p in err["loc"])
            summary_lines.append(f"  [{loc}]  {err['msg']}  (input: {err.get('input')!r})")
        super().__init__("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml_omegaconf(path: str) -> dict[str, Any]:
    """
    Load a YAML file with OmegaConf, resolving ``${...}`` interpolations.

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Plain Python dictionary (OmegaConf internals stripped).

    Raises
    ------
    ImportError
        If OmegaConf is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "OmegaConf is not installed. "
            "Install it with: pip install omegaconf>=2.3.0"
        ) from exc

    cfg = OmegaConf.load(path)
    # to_container resolves all ${} interpolations and returns a plain dict/list
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def _load_yaml_pyyaml(path: str) -> dict[str, Any]:
    """
    Fallback loader using plain PyYAML (no interpolation support).

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Raw parsed YAML as a Python dictionary.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected the YAML file '{path}' to contain a mapping at the top "
            f"level, but got {type(data).__name__}."
        )
    return data


def _read_raw(path: str) -> dict[str, Any]:
    """
    Read a YAML file, preferring OmegaConf and falling back to PyYAML.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration as a plain Python dict.
    """
    try:
        raw = _load_yaml_omegaconf(path)
        logger.debug("Loaded '%s' with OmegaConf (interpolation supported).", path)
    except ImportError:
        logger.warning(
            "OmegaConf not found; falling back to PyYAML "
            "('${...}' interpolations will NOT be resolved)."
        )
        raw = _load_yaml_pyyaml(path)

    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> SimulationConfig:
    """
    Load and validate an ag-cube-cm simulation configuration YAML file.

    This is the **primary entry point** for consuming configuration files.
    The function reads the YAML, resolves OmegaConf variable interpolations
    (e.g. ``${GENERAL_INFO.working_path}``), and validates the result against
    the full :class:`~ag_cube_cm.config.schemas.SimulationConfig` Pydantic model.

    Parameters
    ----------
    path : str | pathlib.Path
        Absolute or relative path to a YAML configuration file.

    Returns
    -------
    SimulationConfig
        A fully validated, typed configuration object.  Access any field with
        IDE auto-completion and zero ``KeyError`` risk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ConfigValidationError
        If the YAML is structurally valid but fails Pydantic validation
        (e.g. missing required keys, wrong types, constraint violations).
    yaml.YAMLError
        If the file content is not valid YAML syntax.

    Examples
    --------
    Typical usage in a simulation script:

    >>> cfg = load_config("configs/maize_dssat_example.yaml")
    >>> cfg.GENERAL_INFO.model
    'dssat'
    >>> cfg.MANAGEMENT.planting_date
    datetime.date(1991, 3, 1)
    >>> cfg.MANAGEMENT.total_n_kg_ha
    300.0

    Building from a dict (useful in tests):

    >>> from ag_cube_cm.config.loader import load_config_from_dict
    >>> cfg = load_config_from_dict({
    ...     "GENERAL_INFO": {"country": "Honduras", "country_code": "HND",
    ...                      "model": "dssat"},
    ...     ...
    ... })
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path.resolve()}'"
        )

    path_str = str(path)
    logger.info("Loading configuration from '%s'.", path_str)

    raw: dict[str, Any] = _read_raw(path_str)

    try:
        config = SimulationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(path_str, exc) from exc

    logger.info(
        "Configuration validated: model=%s, country=%s, feature=%s",
        config.GENERAL_INFO.model,
        config.GENERAL_INFO.country,
        config.SPATIAL_INFO.feature or "ALL",
    )
    return config


def load_config_from_dict(data: dict[str, Any]) -> SimulationConfig:
    """
    Construct a :class:`SimulationConfig` directly from a Python dictionary.

    Useful for programmatic configuration (e.g. in unit tests or notebooks)
    without needing to write a YAML file to disk.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary whose structure mirrors the expected YAML layout.

    Returns
    -------
    SimulationConfig
        Validated configuration object.

    Raises
    ------
    ConfigValidationError
        If the dictionary fails Pydantic validation.

    Examples
    --------
    >>> cfg = load_config_from_dict({
    ...     "GENERAL_INFO": {
    ...         "country": "Malawi",
    ...         "country_code": "MWI",
    ...         "model": "dssat",
    ...     },
    ...     "SPATIAL_INFO": {
    ...         "feature_name": "shapeName",
    ...         "soil_path": "data/soil_mwi.nc",
    ...         "weather_path": "data/weather_mwi.nc",
    ...     },
    ...     "CROP": {"name": "Maize"},
    ...     "MANAGEMENT": {"planting_date": "1991-11-15"},
    ... })
    >>> cfg.GENERAL_INFO.country_code
    'MWI'
    """
    try:
        return SimulationConfig.model_validate(data)
    except ValidationError as exc:
        # Use "<dict>" as the pseudo-path for in-memory configs
        raise ConfigValidationError("<dict>", exc) from exc


def dump_schema(indent: int = 2) -> str:
    """
    Return the JSON Schema for :class:`SimulationConfig` as a formatted string.

    Useful for generating documentation or validating configs with external
    tools (e.g. VS Code YAML extension, ``ajv``).

    Parameters
    ----------
    indent : int
        JSON indentation level.  Defaults to 2.

    Returns
    -------
    str
        Pretty-printed JSON Schema.

    Examples
    --------
    >>> from ag_cube_cm.config.loader import dump_schema
    >>> print(dump_schema())
    {
      "$defs": { ... },
      "properties": { ... },
      ...
    }
    """
    import json

    schema = SimulationConfig.model_json_schema()
    return json.dumps(schema, indent=indent)
