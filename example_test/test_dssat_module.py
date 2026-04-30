"""
Quick smoke-test for the DSSATModel module.

Runs three checks without needing the DSSAT binary:
  1. Config validation via Pydantic
  2. prepare_inputs() → inspect generated .WTH / .SOL / .MZX / DSSBatch files
  3. collect_outputs() with a synthetic Summary.OUT

Run from the project root:
    python test_dssat_module.py
"""

import os
import sys
import textwrap
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Make sure the src layout is on the path when running without install
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ag_cube_cm.config.schemas import (
    CropConfig,
    FertilizerApplication,
    GeneralInfoConfig,
    ManagementConfig,
    SimulationConfig,
    SpatialInfoConfig,
)
from ag_cube_cm.models.dssat.base import DSSATModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, fn):
    try:
        fn()
        print(f"  [{PASS}] {label}")
        return True
    except Exception:
        print(f"  [{FAIL}] {label}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Minimal config
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    return SimulationConfig(
        GENERAL_INFO=GeneralInfoConfig(
            country="TestLand",
            country_code="TST",
            model="dssat",
            working_path="./tmp_dssat_test",
            bin_path=None,
        ),
        SPATIAL_INFO=SpatialInfoConfig(
            feature_name="shapeName",
            soil_path="data/soil.nc",
            weather_path="data/weather.nc",
        ),
        CROP=CropConfig(name="Maize", cultivar="IB1072"),
        MANAGEMENT=ManagementConfig(
            planting_date="2000-03-01",
            fertilizer_schedule=[
                FertilizerApplication(days_after_planting=5, n_kg_ha=100.0),
                FertilizerApplication(days_after_planting=30, n_kg_ha=50.0),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic xarray slices (1 pixel, 365 days)
# ---------------------------------------------------------------------------

def make_weather_slice() -> xr.Dataset:
    dates = pd.date_range("2000-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "tmax": ("date", rng.uniform(25, 35, 365)),
            "tmin": ("date", rng.uniform(15, 25, 365)),
            "solar_radiation": ("date", rng.uniform(10, 25, 365)),
            "precipitation": ("date", rng.uniform(0, 15, 365)),
        },
        coords={
            "date": dates,
            "x": 10.5,
            "y": 5.0,
        },
    )


def make_soil_slice() -> xr.Dataset:
    depths = [10, 20, 40, 60, 100, 200]
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "wv0010": ("depth", rng.uniform(0.35, 0.45, len(depths))),
            "wv0033": ("depth", rng.uniform(0.25, 0.35, len(depths))),
            "wv1500": ("depth", rng.uniform(0.10, 0.20, len(depths))),
            "bdod": ("depth", rng.uniform(1.2, 1.6, len(depths))),
            "soc": ("depth", rng.uniform(0.5, 2.0, len(depths))),
            "clay": ("depth", rng.uniform(20, 40, len(depths))),
            "silt": ("depth", rng.uniform(20, 40, len(depths))),
            "cfvo": ("depth", rng.uniform(0, 5, len(depths))),
        },
        coords={"depth": depths, "x": 10.5, "y": 5.0},
    )


# ---------------------------------------------------------------------------
# Synthetic Summary.OUT
# ---------------------------------------------------------------------------

SUMMARY_OUT = textwrap.dedent("""\
    *SUMMARY : EXPS0001
    !
    @RUNNO TRNO  HWAM  HWAH  BWAH  MDAT  ADAT  PDAT
         1    1  5234  5234  8910   167   145    61
""")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_validation():
    cfg = make_config()
    assert cfg.GENERAL_INFO.model == "dssat"
    assert cfg.CROP.name == "Maize"
    assert len(cfg.MANAGEMENT.fertilizer_schedule) == 2
    assert cfg.MANAGEMENT.total_n_kg_ha == 150.0


def test_prepare_inputs():
    cfg = make_config()
    model = DSSATModel(cfg)
    model.setup_working_directory("pixel_0_0")

    try:
        weather = make_weather_slice()
        soil = make_soil_slice()
        model.prepare_inputs(weather, soil)

        wd = model.working_dir
        generated = [f.name for f in wd.iterdir()]

        assert "WTHE0001.WTH" in generated, f"Missing .WTH — got: {generated}"
        assert "TRAN0001.SOL" in generated, f"Missing .SOL — got: {generated}"
        assert any(f.endswith(".MZX") for f in generated), f"Missing .MZX — got: {generated}"
        assert "DSSBatch.v48" in generated, f"Missing DSSBatch.v48 — got: {generated}"

        # Spot-check .WTH header
        wth = (wd / "WTHE0001.WTH").read_text()
        assert "@DATE  SRAD  TMAX  TMIN  RAIN" in wth

        # Spot-check batch file references the experiment file
        batch = (wd / "DSSBatch.v48").read_text()
        assert "EXPS0001.MZX" in batch

        print(f"        Working dir: {wd}")
        print(f"        Files: {generated}")
    finally:
        model.cleanup_working_directory()


def test_collect_outputs():
    cfg = make_config()
    model = DSSATModel(cfg)
    model.setup_working_directory("pixel_out_test")

    try:
        (model.working_dir / "Summary.OUT").write_text(SUMMARY_OUT)
        result = model.collect_outputs()

        assert "HWAM" in result, f"HWAM missing from result: {result}"
        assert result["HWAM"] == 5234.0, f"Unexpected HWAM: {result['HWAM']}"
        print(f"        Parsed output: {result}")
    finally:
        model.cleanup_working_directory()


def test_crop_code_mapping():
    cfg = make_config()
    model = DSSATModel(cfg)
    cases = {
        "maize": "MZ", "wheat": "WH", "rice": "RI",
        "soybean": "SB", "unknown_crop": "MZ",
    }
    for crop, expected in cases.items():
        got = model._get_crop_code(crop)
        assert got == expected, f"{crop}: expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== DSSATModel smoke tests ===\n")

    results = [
        check("Config validation (Pydantic)",        test_config_validation),
        check("Crop code mapping",                    test_crop_code_mapping),
        check("prepare_inputs -> file generation",    test_prepare_inputs),
        check("collect_outputs -> Summary.OUT parse", test_collect_outputs),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")

    # Clean up temp dir if everything passed
    import shutil
    tmp = Path("./tmp_dssat_test")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    sys.exit(0 if all(results) else 1)
