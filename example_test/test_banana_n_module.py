"""
Smoke-test for the BananaModel (banana_n) module.

Tests:
  1. Config validation via Pydantic
  2. collect_outputs() with injected mock history
  3. collect_outputs() with empty history -> returns {}
  4. Config attribute mismatch diagnostics (silent camelCase fallbacks)
  5. Full prepare_inputs() + run_simulation() end-to-end

Run from the project root:
    python test_banana_n_module.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ag_cube_cm.config.schemas import (
    CropConfig,
    FertilizerApplication,
    GeneralInfoConfig,
    ManagementConfig,
    SimulationConfig,
    SpatialInfoConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(label: str, fn):
    try:
        fn()
        print(f"  [{PASS}] {label}")
        return True
    except AssertionError as exc:
        print(f"  [{FAIL}] {label}")
        print(f"         AssertionError: {exc}")
        return False
    except Exception:
        print(f"  [{FAIL}] {label}")
        traceback.print_exc()
        return False


def warn(label: str, fn):
    try:
        fn()
        print(f"  [{PASS}] {label}")
        return True
    except Exception as exc:
        print(f"  [{WARN}] {label}")
        print(f"         {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Minimal config
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    return SimulationConfig(
        GENERAL_INFO=GeneralInfoConfig(
            country="TestLand",
            country_code="TST",
            model="banana_n",
            working_path="./tmp_banana_test",
            bin_path=None,
        ),
        SPATIAL_INFO=SpatialInfoConfig(
            feature_name="shapeName",
            soil_path="data/soil.nc",
            weather_path="data/weather.nc",
        ),
        CROP=CropConfig(name="Banana", cultivar=None),
        MANAGEMENT=ManagementConfig(
            planting_date="2000-01-01",
            life_cycle_years=1,
            fertilizer_schedule=[
                FertilizerApplication(days_after_planting=30,  n_kg_ha=50.0),
                FertilizerApplication(days_after_planting=100, n_kg_ha=30.0),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic xarray slices
# ---------------------------------------------------------------------------

def make_weather_slice(n_days: int = 730) -> xr.Dataset:
    """
    2 years of daily data.
    - `etr` is required so BanWeather does not try to calculate ETo via
      a missing station method.
    - `date` dimension is what prepare_inputs() renames to DATE.
    """
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "precipitation":   ("date", rng.uniform(0, 10,  n_days)),
            "solar_radiation": ("date", rng.uniform(12, 22, n_days)),
            "tmax":            ("date", rng.uniform(25, 35, n_days)),
            "tmin":            ("date", rng.uniform(15, 25, n_days)),
            "etr":             ("date", rng.uniform(2,  6,  n_days)),
        },
        coords={"date": dates, "x": -88.0, "y": 15.0},
    )


def make_soil_slice() -> xr.Dataset:
    """
    Depths matching the depth_map in prepare_inputs():
      0->0-5, 5->5-15, 15->15-30, 30->30-60
    BanSoil.summarize_depths needs layer 0 (0-30 cm) and layer 1 (30-60 cm).
    All columns accessed by summarize_depths must be present.
    LONG/LAT are added as scalar variables so they survive to_dataframe().
    """
    depths = [0, 5, 15, 30]
    n = len(depths)
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "soc":      ("depth", rng.uniform(1.0,  3.0,  n)),   # -> SOC
            "wv0033":   ("depth", rng.uniform(0.25, 0.35, n)),   # -> fc
            "wv1500":   ("depth", rng.uniform(0.10, 0.20, n)),   # -> pwp
            "clay":     ("depth", rng.uniform(20,   40,   n)),
            "bdod":     ("depth", rng.uniform(1.2,  1.6,  n)),
            "cfvo":     ("depth", rng.uniform(0,    5,    n)),
            "CSOM0":    ("depth", rng.uniform(10,   30,   n)),
            "sand":     ("depth", rng.uniform(20,   40,   n)),
            "nitrogen": ("depth", rng.uniform(0.1,  0.3,  n)),
            "pH":       ("depth", rng.uniform(5.5,  7.0,  n)),
            "LONG":     ("depth", np.full(n, -88.0)),
            "LAT":      ("depth", np.full(n, 15.0)),
        },
        coords={"depth": depths, "x": -88.0, "y": 15.0},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_validation():
    cfg = make_config()
    assert cfg.GENERAL_INFO.model == "banana_n"
    assert cfg.CROP.name == "Banana"
    assert len(cfg.MANAGEMENT.fertilizer_schedule) == 2
    assert cfg.MANAGEMENT.total_n_kg_ha == 80.0


def test_collect_outputs_mock_history():
    """collect_outputs only reads self.history — no simulation needed."""
    from ag_cube_cm.models.banana_n.base import BananaModel
    cfg = make_config()
    model = BananaModel(cfg)
    model.history = [
        {"Avg_SMN_kg_ha": 12.5, "Avg_Bioamass_g_mat": 800.0, "Avg_Fruit_g_mat": 300.0}
        for _ in range(52)
    ]
    result = model.collect_outputs()
    assert "Avg_SMN_kg_ha"   in result
    assert "Avg_Fruit_g_mat" in result
    assert "Yield_kg_ha"     in result
    # Yield = 300 * 1300 / 1000 = 390
    assert result["Yield_kg_ha"] == 390.0, f"Unexpected Yield: {result['Yield_kg_ha']}"
    print(f"        Outputs: {result}")


def test_collect_outputs_empty():
    from ag_cube_cm.models.banana_n.base import BananaModel
    cfg = make_config()
    model = BananaModel(cfg)
    model.history = []
    assert model.collect_outputs() == {}


def test_config_attribute_mismatches():
    """
    Known camelCase vs snake_case mismatches that cause silent fallback to defaults.
    All three attributes below will be None (not found on the Pydantic model),
    so the model silently ignores the config values.
    """
    cfg = make_config()
    issues = []
    if getattr(cfg.MANAGEMENT, "plantingDate",    None) is not None:
        issues.append("plantingDate now exists — model will read planting date correctly")
    if getattr(cfg.MANAGEMENT, "fertilizer",      None) is not None:
        issues.append("fertilizer now exists — model will read fertilizer_schedule correctly")
    if getattr(cfg.MANAGEMENT, "plantingDensity", None) is not None:
        issues.append("plantingDensity now exists — model will read density correctly")

    print("        [INFO] Attribute mismatches (model uses fallback defaults):")
    print("               plantingDate    -> falls back to '2000-01-01'")
    print("               fertilizer      -> falls back to [] (ignores fertilizer_schedule)")
    print("               plantingDensity -> falls back to 1300.0 plants/ha (not in schema)")

    if issues:
        raise AssertionError("Some mismatches have been fixed — update this test: " + "; ".join(issues))


def test_full_pipeline():
    """prepare_inputs + run_simulation + collect_outputs end-to-end."""
    from ag_cube_cm.models.banana_n.base import BananaModel
    cfg = make_config()
    model = BananaModel(cfg)
    model.setup_working_directory("pixel_0_0")

    try:
        weather = make_weather_slice(n_days=730)
        soil    = make_soil_slice()
        model.prepare_inputs(weather, soil)

        assert hasattr(model, "weathert"),            "missing self.weathert"
        assert hasattr(model, "soilt"),               "missing self.soilt"
        assert hasattr(model, "weekly_weather_data"), "missing self.weekly_weather_data"
        assert hasattr(model, "ferti_schedule"),      "missing self.ferti_schedule"
        assert len(model.weekly_weather_data) > 0,   "weekly_weather_data is empty"

        print(f"        nb_weeks:            {model.nb_weeks}")
        print(f"        planting_date used:  {model.planting_date}")
        print(f"        weekly weather rows: {len(model.weekly_weather_data)}")
        print(f"        init_soil_params:    {model.init_soil_params}")

        model.run_simulation()
        assert len(model.history) > 0, "history is empty after run_simulation()"
        print(f"        history weeks:       {len(model.history)}")

        result = model.collect_outputs()
        assert "Yield_kg_ha" in result
        print(f"        Outputs: {result}")

    finally:
        model.cleanup_working_directory()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== BananaModel (banana_n) smoke tests ===\n")

    results = [
        check("Config validation (Pydantic)",          test_config_validation),
        check("collect_outputs with mock history",     test_collect_outputs_mock_history),
        check("collect_outputs with empty history",    test_collect_outputs_empty),
        warn( "Config attribute mismatch diagnostics", test_config_attribute_mismatches),
        check("prepare_inputs + run_simulation (full)", test_full_pipeline),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\n{passed}/{total} passed")

    import shutil
    tmp = Path("./tmp_banana_test")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    sys.exit(0 if all(results) else 1)
