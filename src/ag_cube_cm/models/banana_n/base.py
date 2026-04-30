"""
ag_cube_cm.models.banana_n.base
================================

Banana_N CropModel Implementation.
"""

import logging
from typing import Any, Dict

import xarray as xr
import pandas as pd
from datetime import datetime

from ag_cube_cm.models.base import CropModel
from ag_cube_cm.models.factory import register_model

from ._base import BANANAField
from .weather import BanWeather
from .soil import BanSoil

logger = logging.getLogger(__name__)

@register_model("banana_n")
class BananaModel(CropModel):
    """
    Banana_N Implementation of the CropModel ABC.
    
    This is a pure Python model. It does not require writing files to disk
    nor does it invoke any subprocesses. Everything is executed in-memory.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.history = []

    def prepare_inputs(self, weather_slice: xr.Dataset, soil_slice: xr.Dataset) -> None:
        """
        Extract Xarray slices into Pandas DataFrames and initialize
        the model's internal data structures.
        """
        df_wth = weather_slice.to_dataframe().reset_index().dropna()
        df_sol = soil_slice.to_dataframe().reset_index().dropna()

        if df_sol.empty or df_wth.empty:
            raise ValueError("Empty slices passed to Banana_N.")

        # 1. Weather Setup
        lat = float(df_sol['y'].iloc[0]) if 'y' in df_sol.columns else 0.0
        lon = float(df_sol['x'].iloc[0]) if 'x' in df_sol.columns else 0.0

        # Rename to BanWeather expected names (skip if already correct)
        wth_rename = {}
        if 'precipitation' in df_wth.columns:
            wth_rename['precipitation'] = 'rain'
        if 'solar_radiation' in df_wth.columns:
            wth_rename['solar_radiation'] = 'srad'
        if wth_rename:
            df_wth = df_wth.rename(columns=wth_rename)
        if 'date' in df_wth.columns:
            df_wth = df_wth.rename(columns={'date': 'DATE'})
        df_wth['DATE'] = pd.to_datetime(df_wth['DATE']).dt.strftime('%Y%m%d')

        # Convert temperatures from Kelvin to Celsius if stored as Kelvin (AgERA5 uses K)
        for temp_col in ['tmin', 'tmax']:
            if temp_col in df_wth.columns and df_wth[temp_col].mean() > 100:
                df_wth[temp_col] = df_wth[temp_col] - 273.15

        # Convert solar radiation from J/m²/d to MJ/m²/d if stored in Joules (AgERA5 uses J m-2 d-1)
        if 'srad' in df_wth.columns and df_wth['srad'].mean() > 10000:
            df_wth['srad'] = df_wth['srad'] / 1e6

        self.weathert = BanWeather(df=df_wth, latitude=lat, longitude=lon, altitude=0)

        # 2. Soil Setup
        df_sol = df_sol.rename(columns={'soc': 'SOC', 'wv0033': 'fc', 'wv1500': 'pwp'})
        if 'phh2o' in df_sol.columns and 'pH' not in df_sol.columns:
            df_sol = df_sol.rename(columns={'phh2o': 'pH'})

        # Depth: handle both numeric (synthetic) and string (real datacubes)
        if 'depth' in df_sol.columns:
            first_val = str(df_sol['depth'].iloc[0])
            if '-' in first_val:
                df_sol['DEPTH'] = df_sol['depth'].astype(str)
            else:
                depth_map = {0: "0-5", 5: "5-15", 15: "15-30",
                             30: "30-60", 60: "60-100", 100: "100-200"}
                df_sol['DEPTH'] = df_sol['depth'].map(
                    lambda x: depth_map.get(int(x), f"{int(x)}-{int(x)+10}")
                )

        # Derive columns needed by BanSoil that may be absent
        if 'CSOM0' not in df_sol.columns and 'SOC' in df_sol.columns:
            df_sol['CSOM0'] = df_sol['SOC']
        if 'LONG' not in df_sol.columns:
            df_sol['LONG'] = lon
        if 'LAT' not in df_sol.columns:
            df_sol['LAT'] = lat

        self.soilt = BanSoil(df=df_sol)

        # 3. Time parameters — use snake_case Pydantic fields
        planting_date_val = getattr(self.config.MANAGEMENT, 'planting_date', None)
        self.planting_date = (
            str(planting_date_val)
            if planting_date_val is not None
            else '2000-01-01'
        )
        life_cycle_years = (
            getattr(self.config.MANAGEMENT, 'life_cycle_years', None) or
            getattr(self.config.GENERAL_INFO, 'number_years', 1)
        )
        self.nb_weeks = (life_cycle_years or 1) * 52

        self.init_soil_params = self._calculate_soil_initial_conditions()

        end_date = pd.to_datetime(self.planting_date) + pd.Timedelta(weeks=self.nb_weeks + 1)
        self.weekly_weather_data = self.weathert.weekly_weather(
            starting_date=self.planting_date,
            ending_date=end_date.strftime('%Y-%m-%d')
        ).to_dict(orient='records')

        # 4. Fertilizer — use snake_case field name
        fert_apps = getattr(self.config.MANAGEMENT, 'fertilizer_schedule', []) or []
        self.ferti_schedule = [
            {'application': False, 'q_org': 0.0, 'min_f': 0.0}
            for _ in range(self.nb_weeks)
        ]
        for fert in fert_apps:
            dap = getattr(fert, 'days_after_planting', 0)
            week_idx = dap // 7
            if 0 <= week_idx < self.nb_weeks:
                n_amt = getattr(fert, 'n_kg_ha', getattr(fert, 'n_amount', 0.0))
                self.ferti_schedule[week_idx] = {'application': True, 'q_org': 0.0, 'min_f': n_amt}

    def _calculate_soil_initial_conditions(self):
        """Estimate starting conditions using 90-days pre-planting weather."""
        def layer_properties(layer_idx):
            layerprts = self.soilt.get_son(layer_idx)
            smn = self.soilt.get_initial_smn(
                self.weathert, 
                self.planting_date,
                layerprts['son'],
                layerprts['clay'], 
                layer_depth_cm=self.soilt.depths[layer_idx][1] - self.soilt.depths[layer_idx][0]
            )
            return layerprts, smn

        try:
            layer_1, smn1 = layer_properties(0)
            layer_2, smn2 = layer_properties(1)
        except Exception as e:
            logger.warning(f"Using default soil initialization due to missing/invalid layers: {e}")
            return {
                'wsol1': 100.0, 'wsol2': 100.0, 
                'son': 3000.0, 'smn_depth1': 20.0, 'smn_depth2': 10.0
            }

        return {
            'wsol1': layer_1['Wsol'], 'wsol2': layer_2['Wsol'], 
            'son': (layer_1['son']+ layer_2['son'])/2, 'smn_depth1': smn1, 'smn_depth2': smn2
        }

    def run_simulation(self) -> None:
        """
        Execute the Banana_N python simulation in memory.
        """
        nban = 40
        density = getattr(self.config.MANAGEMENT, 'plantingDensity', 1300.0)
        
        field = BANANAField(nban=nban, density=density, init_soil_parameters=self.init_soil_params)
        
        self.history = field.simulate(
            nb_weeks=self.nb_weeks,
            weather_data=self.weekly_weather_data,
            ferti_schedule=self.ferti_schedule
        )

    def collect_outputs(self) -> Dict[str, Any]:
        """
        Return the final state of the simulation as the yield result.
        """
        if not self.history:
            return {}
            
        last_state = self.history[-1]
        
        return {
            'Avg_SMN_kg_ha': last_state.get('Avg_SMN_kg_ha', 0.0),
            'Avg_Biomass_g_mat': last_state.get('Avg_Bioamass_g_mat', 0.0),
            'Avg_Fruit_g_mat': last_state.get('Avg_Fruit_g_mat', 0.0),
            # Calculate total Yield (kg/ha)
            'Yield_kg_ha': (last_state.get('Avg_Fruit_g_mat', 0.0) * getattr(self.config.MANAGEMENT, 'plantingDensity', 1300.0)) / 1000.0
        }
