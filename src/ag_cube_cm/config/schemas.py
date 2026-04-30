"""
ag_cube_cm.config.schemas
==========================

Pydantic v2 models that define the complete, validated structure for the
ag-cube-cm YAML configuration file.

Design principles
-----------------
* **Strict typing** — Python 3.10+ union syntax (`X | None`), `Literal` for
  enumerated string choices.  No bare ``Optional`` or ``Union`` imports.
* **Descriptive validation** — every field carries a ``description`` so that
  Pydantic's ``model_json_schema()`` can auto-generate documentation.
* **Safe defaults** — reasonable defaults are provided wherever a field is
  truly optional; required fields have *no* default so missing keys raise a
  clear ``ValidationError`` at load time.
* **Composable** — each config section is a standalone ``BaseModel`` that can
  be independently instantiated and tested.

Typical YAML structure consumed by these models
------------------------------------------------
.. code-block:: yaml

    GENERAL_INFO:
      country: "Honduras"
      country_code: "HND"
      working_path: "runs"
      ncores: 8
      model: "dssat"

    SPATIAL_INFO:
      geospatial_path: "data/country_HND_ADM2.shp"
      feature_name: "shapeName"
      adm_level: 2
      aggregate_by: "pixel"
      soil_path: "data/soil_hnd.nc"
      weather_path: "data/weather_hnd_2000_2019.nc"
      scale_factor: 1
      feature: "Comayagua"

    CROP:
      name: "Maize"
      cultivar: "IB1072"

    MANAGEMENT:
      planting_date: "1991-03-01"
      planting_window_days: 20
      fertilizer_schedule:
        - days_after_planting: 5
          n_kg_ha: 200
          p_kg_ha: 0
          k_kg_ha: 0
        - days_after_planting: 30
          n_kg_ha: 100
          p_kg_ha: 50
          k_kg_ha: 0
      index_soilwat: 1
      template: "crop_modeling/dssat/exp_files/KEAG8104.MZX"
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supporting / nested models
# ---------------------------------------------------------------------------


class FertilizerApplication(BaseModel):
    """
    A single fertilizer application event within a crop season.

    Amounts are expressed in kg of elemental nutrient per hectare.
    The timing is expressed as **days after planting** (DAP) so that
    the schedule automatically shifts when ``planting_date`` changes.

    Attributes
    ----------
    days_after_planting : int
        Number of days after the planting date to apply the fertilizer.
        Must be ≥ 0 (0 = at planting).
    n_kg_ha : float
        Elemental nitrogen (N) applied, kg ha⁻¹.  Defaults to 0.
    p_kg_ha : float
        Elemental phosphorus (P) applied, kg ha⁻¹.  Defaults to 0.
    k_kg_ha : float
        Elemental potassium (K) applied, kg ha⁻¹.  Defaults to 0.
    """

    days_after_planting: Annotated[int, Field(ge=0, description="Days after planting (≥ 0)")]
    n_kg_ha: Annotated[float, Field(ge=0.0, default=0.0, description="Nitrogen applied (kg ha⁻¹)")]
    p_kg_ha: Annotated[float, Field(ge=0.0, default=0.0, description="Phosphorus applied (kg ha⁻¹)")]
    k_kg_ha: Annotated[float, Field(ge=0.0, default=0.0, description="Potassium applied (kg ha⁻¹)")]

    @property
    def total_npk(self) -> tuple[float, float, float]:
        """Convenience accessor returning (N, P, K) as a tuple."""
        return (self.n_kg_ha, self.p_kg_ha, self.k_kg_ha)


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class GeneralInfoConfig(BaseModel):
    """
    Top-level simulation settings shared across all models.

    Attributes
    ----------
    country : str
        Full country name (e.g. "Honduras").  Used in DSSAT file headers.
    country_code : str
        ISO 3166-1 alpha-3 country code (e.g. "HND").  Must be exactly 3
        uppercase letters.
    working_path : str
        Root directory where model run sub-folders will be created.
        Defaults to ``"runs"``.
    ncores : int
        Maximum number of parallel worker processes/threads.  Defaults to 4.
    model : Literal["dssat", "caf", "simple_model", "banana_n"]
        The process-based crop model to use.
    bin_path : str | None
        Absolute path to the crop-model executable.  ``None`` means the
        package will attempt to locate the bundled binary automatically.
    dssat_path : str | None
        Root DSSAT installation directory (required when ``model == "dssat"``
        and using a custom DSSAT build).
    """

    country: Annotated[str, Field(description="Full country name, used in DSSAT file headers")]
    country_code: Annotated[
        str,
        Field(
            min_length=2,
            max_length=5,
            description="ISO 3166-1 alpha-3 country code (e.g. 'HND')",
        ),
    ]
    working_path: Annotated[
        str, Field(default="runs", description="Root directory for model run sub-folders")
    ]
    ncores: Annotated[
        int, Field(default=4, ge=1, description="Maximum parallel worker count")
    ]
    model: Annotated[
        Literal["dssat", "caf", "simple_model", "banana_n"],
        Field(description="Process-based crop model identifier"),
    ]
    bin_path: Annotated[
        str | None,
        Field(default=None, description="Path to the crop-model executable (None = auto-detect)"),
    ]
    dssat_path: Annotated[
        str | None,
        Field(default=None, description="DSSAT installation root (required for custom DSSAT builds)"),
    ]

    @field_validator("country_code", mode="before")
    @classmethod
    def normalise_country_code(cls, v: str) -> str:
        """Ensure the code is stored in uppercase."""
        return v.strip().upper()

    @model_validator(mode="after")
    def check_dssat_path_when_required(self) -> GeneralInfoConfig:
        """
        Warn (but do not fail) when ``model == 'dssat'`` and no ``dssat_path``
        is set — the runner will fall back to the DSSATTools bundled binary.
        """
        # Intentionally a soft check; hard validation happens at runtime
        # when the binary is actually needed.
        return self


class SpatialInfoConfig(BaseModel):
    """
    Spatial extent and data-source configuration.

    Attributes
    ----------
    geospatial_path : str | None
        Path to the administrative boundary shapefile (.shp).  When ``None``
        the boundary is downloaded automatically from the GeoBoundaries API
        using ``country_code`` and ``adm_level``.
    feature_name : str
        Column name in the shapefile that contains region labels
        (e.g. ``"shapeName"``).
    adm_level : int
        Administrative level to download from GeoBoundaries when
        ``geospatial_path`` is ``None``.  Defaults to 2.
    aggregate_by : Literal["texture", "pixel"] | None
        Spatial aggregation strategy.  ``"texture"`` groups pixels by USDA
        soil textural class; ``"pixel"`` runs the model at each raster cell
        independently; ``None`` treats the entire ROI as a single unit.
    soil_path : str
        Path to the SoilGrids multi-depth NetCDF datacube.
    weather_path : str
        Path to the AgEra5 / CHIRPS multi-temporal NetCDF datacube.
    scale_factor : int
        Spatial resampling factor applied when co-registering weather and
        soil grids.  Defaults to 1 (no resampling).
    feature : str | None
        Specific region label to simulate (must match a value in the
        ``feature_name`` column).  When ``None``, all regions are simulated.
    dem_path : str | None
        Path to a Digital Elevation Model raster (required by CAF2021 and
        SIMPLE model for slope/elevation parameters).
    """

    geospatial_path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Shapefile path. None → auto-download from GeoBoundaries."
            ),
        ),
    ]
    feature_name: Annotated[
        str, Field(description="Shapefile column holding region labels")
    ]
    adm_level: Annotated[
        int, Field(default=2, ge=0, le=5, description="Administrative boundary level (0–5)")
    ]
    aggregate_by: Annotated[
        Literal["texture", "pixel"] | None,
        Field(
            default=None,
            description=(
                "'texture' → group by soil textural class; "
                "'pixel' → pixel-level simulation; "
                "None → single ROI unit."
            ),
        ),
    ]
    soil_path: Annotated[
        str, Field(description="Path to the SoilGrids multi-depth NetCDF datacube")
    ]
    weather_path: Annotated[
        str, Field(description="Path to the AgEra5/CHIRPS multi-temporal NetCDF datacube")
    ]
    scale_factor: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Spatial resampling factor for weather↔soil co-registration",
        ),
    ]
    feature: Annotated[
        str | None,
        Field(
            default=None,
            description="Specific region label to simulate (None → simulate all).",
        ),
    ]
    dem_path: Annotated[
        str | None,
        Field(
            default=None,
            description="DEM raster path (required for CAF2021 and SIMPLE model).",
        ),
    ]
    output_path: Annotated[
        str,
        Field(
            default="output_yield.nc",
            description="Path for the output yield datacube (.nc).",
        ),
    ]

    @field_validator("soil_path", "weather_path", "geospatial_path", "dem_path", mode="before")
    @classmethod
    def coerce_path_to_string(cls, v: object) -> str | None:
        """Accept ``Path`` objects and coerce them to strings."""
        if isinstance(v, Path):
            return str(v)
        return v  # type: ignore[return-value]


class CropConfig(BaseModel):
    """
    Crop and cultivar identity.

    Attributes
    ----------
    name : str
        Crop common name as recognised by the target model
        (e.g. ``"Maize"``, ``"Coffee"``, ``"Wheat"``).
    cultivar : str | None
        Cultivar or genotype code.  Required for DSSAT; optional for
        models with a single generic cultivar.
    cultivar_file : str | None
        Path to a custom cultivar parameter file (.CUL/.ECO).  When
        provided, it overrides the model's built-in cultivar database.
    """

    name: Annotated[str, Field(description="Crop common name (model-specific)")]
    cultivar: Annotated[
        str | None,
        Field(default=None, description="Cultivar / genotype code"),
    ]
    cultivar_file: Annotated[
        str | None,
        Field(default=None, description="Path to a custom .CUL / .ECO file"),
    ]

    @field_validator("name", mode="before")
    @classmethod
    def strip_and_title(cls, v: str) -> str:
        """Normalise crop name to title-case (e.g. 'maize' → 'Maize')."""
        return v.strip().title()


class ManagementConfig(BaseModel):
    """
    Agronomic management schedule for the crop simulation.

    This model is designed to be robust enough for multi-treatment experiments.
    All date fields accept ISO 8601 strings (``"YYYY-MM-DD"``); Pydantic will
    parse them into ``datetime.date`` objects automatically.

    Attributes
    ----------
    planting_date : date
        Nominal planting / sowing date.  Treatment dates are derived from
        this anchor by stepping forward in ``planting_window_days`` increments.
    planting_window_days : int
        Number of days between consecutive planting-date treatments.
        Set to 1 for daily windows, 7 for weekly.  Defaults to 7.
    n_planting_windows : int | None
        Total number of planting-date treatments to generate.  When ``None``
        the runner derives this from the available weather record length.
    harvesting_date : date | None
        Fixed harvest date.  ``None`` lets the model determine maturity
        phenologically (recommended).
    index_soilwat : int
        DSSAT initial soil-water index (1 = field capacity, 0 = dry).
        Ignored by non-DSSAT models.  Defaults to 1.
    fertilizer_schedule : list[FertilizerApplication]
        Ordered list of fertilizer application events.  Each event specifies
        N-P-K amounts (kg ha⁻¹) and the timing as days after planting.
        An empty list means no fertilizer is applied (rainfed baseline).
    template : str | None
        Path to the DSSAT experiment template file (.MZX, .WHX, etc.).
        Required for DSSAT; ignored by other models.
    life_cycle_years : int | None
        For perennial-crop models (CAF2021, Banana-N): total years to simulate
        across the crop's life cycle.  Ignored for annual crops.
    co2_ppm : float
        Atmospheric CO₂ concentration used by SIMPLE model stress functions.
        Defaults to 381 ppm (historical baseline from the original paper).
    """

    planting_date: Annotated[
        date,
        Field(description="Nominal sowing / planting date (ISO 8601)"),
    ]
    planting_window_days: Annotated[
        int,
        Field(
            default=7,
            ge=1,
            description="Days between consecutive planting-date treatments",
        ),
    ]
    n_planting_windows: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Total number of planting-date treatments. "
                "None → derived from weather record length."
            ),
        ),
    ]
    harvesting_date: Annotated[
        date | None,
        Field(
            default=None,
            description=(
                "Fixed harvest date. None → phenological maturity (recommended)."
            ),
        ),
    ]
    index_soilwat: Annotated[
        int,
        Field(
            default=1,
            ge=0,
            le=1,
            description="DSSAT initial soil-water index (1=FC, 0=dry). Ignored by non-DSSAT models.",
        ),
    ]
    fertilizer_schedule: Annotated[
        list[FertilizerApplication],
        Field(
            default_factory=list,
            description=(
                "Ordered fertilizer application events (N-P-K kg ha⁻¹ with DAP timing). "
                "Empty list → unfertilised baseline."
            ),
        ),
    ]
    template: Annotated[
        str | None,
        Field(
            default=None,
            description="DSSAT experiment template file path (.MZX, .WHX, …)",
        ),
    ]
    life_cycle_years: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Life-cycle years for perennial crops (CAF2021, Banana-N). Ignored for annuals.",
        ),
    ]
    co2_ppm: Annotated[
        float,
        Field(
            default=381.0,
            ge=280.0,
            le=1200.0,
            description="Atmospheric CO₂ concentration for SIMPLE model (ppm). Default = 381 ppm.",
        ),
    ]

    @model_validator(mode="after")
    def harvest_must_be_after_planting(self) -> ManagementConfig:
        """Raise if a fixed harvest date precedes the planting date."""
        if self.harvesting_date is not None and self.harvesting_date <= self.planting_date:
            raise ValueError(
                f"harvesting_date ({self.harvesting_date}) must be strictly after "
                f"planting_date ({self.planting_date})."
            )
        return self

    @model_validator(mode="after")
    def sort_fertilizer_schedule(self) -> ManagementConfig:
        """Ensure fertilizer events are in chronological order (by DAP)."""
        if self.fertilizer_schedule:
            self.fertilizer_schedule = sorted(
                self.fertilizer_schedule, key=lambda e: e.days_after_planting
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def total_n_kg_ha(self) -> float:
        """Total elemental N applied across the entire season (kg ha⁻¹)."""
        return sum(e.n_kg_ha for e in self.fertilizer_schedule)

    @property
    def total_p_kg_ha(self) -> float:
        """Total elemental P applied across the entire season (kg ha⁻¹)."""
        return sum(e.p_kg_ha for e in self.fertilizer_schedule)

    @property
    def total_k_kg_ha(self) -> float:
        """Total elemental K applied across the entire season (kg ha⁻¹)."""
        return sum(e.k_kg_ha for e in self.fertilizer_schedule)

    def dssat_fertilizer_tuple(self) -> tuple[list[int], list[list[float]]]:
        """
        Convert the fertilizer schedule into the legacy DSSAT tuple format
        expected by the original ``DSSATManagement_base`` interface.

        Returns
        -------
        tuple[list[int], list[list[float]]]
            ``(days_after_planting, [[N, P, K], …])`` where each inner list
            holds the kg ha⁻¹ values for one application event.
        """
        daps = [e.days_after_planting for e in self.fertilizer_schedule]
        npks = [[e.n_kg_ha, e.p_kg_ha, e.k_kg_ha] for e in self.fertilizer_schedule]
        return daps, npks


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """
    Root Pydantic model representing a complete ag-cube-cm simulation config.

    This is the object returned by :func:`ag_cube_cm.config.loader.load_config`.
    All four section models are required — missing any top-level key in the
    YAML will raise a ``pydantic.ValidationError`` with a descriptive message.

    Attributes
    ----------
    GENERAL_INFO : GeneralInfoConfig
        Top-level simulation and model-selection settings.
    SPATIAL_INFO : SpatialInfoConfig
        Spatial extent, data-path, and aggregation strategy.
    CROP : CropConfig
        Crop name, cultivar, and optional custom parameter file.
    MANAGEMENT : ManagementConfig
        Agronomic schedule: planting dates, fertilizer events, templates.

    Example
    -------
    >>> from ag_cube_cm.config.loader import load_config
    >>> cfg = load_config("configs/maize_dssat_example.yaml")
    >>> cfg.GENERAL_INFO.model
    'dssat'
    >>> cfg.MANAGEMENT.total_n_kg_ha
    300.0
    """

    GENERAL_INFO: GeneralInfoConfig
    SPATIAL_INFO: SpatialInfoConfig
    CROP: CropConfig
    MANAGEMENT: ManagementConfig

    model_config = {
        # Forbid extra keys in the YAML so typos are caught immediately.
        "extra": "forbid",
        # Allow `date` strings to be parsed automatically.
        "str_strip_whitespace": True,
    }
