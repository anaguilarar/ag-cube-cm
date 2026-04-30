### son = n * bd * t * 1 - cf https://cdnsciencepub.com/doi/epdf/10.4141/cjss95-075

## PMN fraction of organic matter conver to plant available N (ammonium and nitrate) https://www.nrcs.usda.gov/sites/default/files/2022-10/potentially_mineralizable_nitrogen.pdf
#The rate of mineralization depends on soil texture (clay protects organic matter), temperature, moisture, and soil microbial activity.
# inital smn = (son * krothc -> temperature) * r rothc -> moisture

## RothC rate-modifying factors


# ┌──────────────────────┬────────────────────┬───────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
# │ Component            │ Input Type         │ Role in Final SMN0                                            │ Scientific Justification                                                                          │
# ├──────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
# │ Averaged Temperature │ 90-day mean        │ Drives the biological speed of N release (fT)                 │ Mineralization follows first-order kinetics; rate constant k is exponentially temperature-driven  │
# │ Precipitation        │ 30-day sum         │ Determines availability and retention (fW)                    │ Soil moisture controls microbial activity and N movement (retention vs. leaching)                 │
# │ Clay %               │ 0–100              │ Water-holding capacity & protection                           │ Clay protects organic matter and defines SMDmax used in RothC moisture logic                      │
# │ 0.0026               │ Constant           │ 13 weeks of potential mineralization                          │ From weekly Ksom = 0.0002 for tropical andosols (0.0002 × 13 ≈ 0.0026)                            │
# │ 0.007                │ Constant           │ Background mineral pool ratio                                 │ Empirical SMN:SON ratio from banana system initialization data                                    │
# └──────────────────────┴────────────────────┴───────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘

# ┌────────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
# │ Variable           │ Description                                  │ Equation / Constant                                                          │ Reference                                                  │
# ├────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
# │ N_min              │ Newly mineralized nitrogen                   │ N_min = (SON * 0.0026) * f_T * f_W                                           │ Ruillé et al. (2025) (BANANA-N initialization)             │
# │ N_residual         │ Stable background mineral N                  │ N_residual = 0.005 * SON                                                     │ Derived from experimental andosol data                     │
# │ f_T                │ Temperature modifier (RMF_Tmp)               │ f_T = 47.91 / (exp(106.06 / (T + 18.27)) + 1)                                │ Weihermüller et al. (2013) (RothC-26.3)                    │
# │ f_W                │ Moisture modifier (RMF_Moist)                │ f_W = 0.2 + 0.8 * (max_smd - acc_tsmd) / (max_smd - threshold)               │ Weihermüller et al. (2013); Jenkinson et al. (1990)        │
# │ SON                │ Soil Organic Nitrogen Stock                  │ SON = N * BD * T * (1 - CF) * 100                                            │ Ellert & Bettany (1995)                                    │
# └────────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘



import numpy as np
from typing import Tuple, Optional, Dict, Any, Union 
import pandas as pd
from datetime import datetime, timedelta

def RMF_Moist(precip: np.ndarray, evaporation: np.ndarray, soil_thickness: float, 
              clay: float, bare: Optional[np.ndarray], pE: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the RothC moisture rate-modifying factor (f_W) and moisture deficit.
    
    Parameters
    ----------
    precip : numpy.ndarray
        Precipitation time series (mm).
    evaporation : numpy.ndarray
        Potential evapotranspiration time series (mm).
    soil_thickness : float
        Thickness of the soil layer (cm).
    clay : float
        Clay content percentage (0-100).
    bare : numpy.ndarray, optional
        Boolean array where True indicates bare soil periods.
    pE : float, optional
        Evaporation coefficient (standard 0.75), by default 0.75.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        - acc_TSMD: Accumulated topsoil moisture deficit (mm).
        - b: Moisture rate-modifying factor (f_W).

    References
    ----------
    Weihermüller et al. (2013); Jenkinson et al. (1990).
    """
    # Constants from RothC-26.3    
    MAX_SMD_CONSTANTS = [20, 1.3, 0.01, 23.0]
    B_VALUE_CONSTANTS = [0.2, 0.8, 0.444] # [min_moisture, range, threshold_fraction]
    
    # 1. Calculate Maximum SMD 
    max_smd_value = -(
        MAX_SMD_CONSTANTS[0]
        + MAX_SMD_CONSTANTS[1] * clay
        - MAX_SMD_CONSTANTS[2] * clay**2
    ) * (soil_thickness / MAX_SMD_CONSTANTS[3])
    
    max_smd = np.full(evaporation.shape[0], max_smd_value)
    if bare is not None:
        max_smd[bare.astype(bool)] /= 1.8

    # 2. Calculate accumulated deficit (acc_TSMD)
    M = precip - evaporation * pE
    acc_TSMD = np.zeros_like(M, dtype=np.float64)
    acc_TSMD[0] = min(M[0], 0)
    for i in range(1, len(M)):
        acc_TSMD[i] = max(min(acc_TSMD[i - 1] + M[i], 0), max_smd[i])

    # 3. Calculate b (Moisture Rate Modifier) 
    threshold = B_VALUE_CONSTANTS[2] * max_smd
    
    # THIS IS THE MISSING LINE: Calculate the stress factor for the entire array first.
    # This formula creates a linear ramp from 0.2 (at max dryness) to 1.0 (at the threshold).
    b = B_VALUE_CONSTANTS[0] + B_VALUE_CONSTANTS[1] * (max_smd - acc_TSMD) / (max_smd - threshold)
    
    # Now, set b to 1.0 for any day the soil is wetter than the threshold.
    b[acc_TSMD > threshold] = 1.0
    
    # Ensure b does not go below the minimum of 0.2
    b = np.maximum(b, B_VALUE_CONSTANTS[0])
    
    return acc_TSMD, b

def RMF_Tmp(temperature: np.ndarray) -> np.ndarray:
    """
    Calculate the RothC temperature rate-modifying factor (f_T).

    Parameters
    ----------
    temperature : numpy.ndarray
        Mean air/soil temperature time series (°C).

    Returns
    -------
    numpy.ndarray
        Temperature rate-modifying factor (f_T).

    References
    ----------
    Weihermüller et al. (2013) (RothC-26.3).
    """
    STRESS_COEF_CONSTANTS = [47.91, 106.06, 18.27]
    
    stress_coef = STRESS_COEF_CONSTANTS[0] / (
            1
            + np.exp(
                STRESS_COEF_CONSTANTS[1]
                / (temperature + STRESS_COEF_CONSTANTS[2])
            )
        )
    stress_coef[temperature < -STRESS_COEF_CONSTANTS[2]] = np.nan
    return stress_coef
    

class NitrogenMineralization:
    """
    Calculator for Soil Organic Nitrogen (SON) and its mineralization over time.

    This class integrates soil properties and weather data to estimate the 
    mineral nitrogen (SMN0) available for crop uptake based on RothC logic.
    """

    def __init__(self) -> None:
        """
        Initialize the NitrogenMineralization object.
        """
        pass
    
    def calculate_son(self, soil_nitrogen: float, soil_depth: float, soil_bulk_density: float, soil_cf: float) -> float:
        """
        Calculate the Soil Organic Nitrogen (SON) stock.

        Parameters
        ----------
        soil_nitrogen : float
            Total soil nitrogen content (%).
        soil_depth : float
            Thickness of the soil layer (cm).
        soil_bulk_density : float
            Soil bulk density (g/cm³).
        soil_cf : float
            Coarse fragment volume fraction (0-1).

        Returns
        -------
        float
            The SON stock in kg/ha.

        References
        ----------
        Ellert & Bettany (1995) standard conversion: SON = N * BD * T * (1 - CF) * 100.
        """
        return soil_nitrogen * soil_depth * soil_bulk_density * (1 - soil_cf) * 100

    def calculateSM0(self, son: float, temperature: np.ndarray, rain: np.ndarray, 
                     evap: np.ndarray, clay: float, soil_depth: float, 
                     bare: Optional[np.ndarray] = None, pE: float = 0.75, 
                     daily_value: bool = False) -> Union[float, np.ndarray]:
        """
        Calculate the potential mineral nitrogen (SMN0) available for the crop.

        Integrates the SON stock with temperature and moisture rate-modifying 
        factors (RMF) over a specific period.

        Parameters
        ----------
        son : float
            Soil Organic Nitrogen stock (kg/ha).
        temperature : numpy.ndarray
            Mean air/soil temperature time series (°C).
        rain : numpy.ndarray
            Precipitation time series (mm).
        evap : numpy.ndarray
            Potential evapotranspiration time series (mm).
        clay : float
            Clay content percentage (0-100 or 0-1 fraction).
        soil_depth : float
            Soil layer thickness (cm).
        bare : numpy.ndarray, optional
            Boolean array for bare soil periods. If None, assumes no bare soil.
        pE : float, optional
            Evaporation coefficient, by default 0.75.
        daily_value : bool, optional
            If True, returns the daily mineralization series. If False, returns 
             the total mineralization for the period, by default False.

        Returns
        -------
        Union[float, numpy.ndarray]
            Potential mineral nitrogen available (kg/ha). 
            Returns a scalar if daily_value is False, otherwise an array.
        """
        clay_pct = clay * 100 if clay <= 1.0 else clay
        if bare is None:
            bare = np.zeros_like(temperature)

        f_T = RMF_Tmp(temperature) 

        _, f_W  = RMF_Moist(rain, evap, soil_depth, clay_pct, bare, pE=pE)

        if not daily_value:

            f_T = np.mean(f_T)
            f_W = np.mean(f_W)

        N_min  = (son * 0.0026) * f_T * f_W 
        N_residual = 0.003 * son 
        
        return N_min + N_residual
    
###


class BanSoil(NitrogenMineralization):
    """
    Soil management class for Banana crop modeling, extending NitrogenMineralization.

    Handles soil property aggregation across depths and initial water state calculations.
    """

    @property
    def soil(self) -> pd.DataFrame:
        """
        The soil data loaded from a CSV or provided as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The underlying soil property data.
        """
        if self._soil is None and self.path is not None:
            self._soil = pd.read_csv(self.path)
        
        return self._soil

    def __init__(self, path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize the BanSoil manager.

        Parameters
        ----------
        path : str, optional
            Path to the soil CSV file.
        df : pd.DataFrame, optional
            Existing DataFrame with soil properties.
        """
        super().__init__()
        self.path = path
        self._soil = None
        self.depths = [[0, 30], [30, 60]]  # Ruille 2025 Layer 1 (0-30), Layer 2 (30-60)

        if df is not None:
            self._soil = df
    
    def summarize_depths(self, layer: int) -> Dict[str, Any]:
        """
        Aggregates soil properties for a specific layer index using depth-weighted averaging.

        Parameters
        ----------
        layer : int
            The layer index:
            - 0 -> [0, 30] cm
            - 1 -> [30, 60] cm

        Returns
        -------
        Dict[str, Any]
            Dictionary containing depth-weighted soil properties (SOC, clay, nitrogen, etc.).

        Raises
        ------
        ValueError
            If no soil data is found for the specified depth range.
        """
        target_top, target_bottom = self.depths[layer]
        df = self.soil.copy()

        df[['top', 'bottom']] = df['DEPTH'].str.split('-', expand=True).astype(float)
        
        mask = (df['top'] >= target_top) & (df['bottom'] <= target_bottom)
        layer_df = df[mask].copy()
        
        if layer_df.empty:
            raise ValueError(f"No soil data found for depth range {target_top}-{target_bottom} cm.")
        
        layer_df['thickness'] = layer_df['bottom'] - layer_df['top']
        total_thickness = layer_df['thickness'].sum()
        
        properties_to_average = ['SOC', 'CSOM0', 'clay', 'sand', 'nitrogen', 'pH', 'bdod', 'cfvo', 'fc', 'pwp']
        
        summary = {}
        for prop in properties_to_average:
            # (Value * Thickness) / Total Thickness
            weighted_sum = (layer_df[prop] * layer_df['thickness']).sum()
            summary[prop] = weighted_sum / total_thickness
            
        summary['layer_index'] = layer
        summary['top_cm'] = target_top
        summary['bottom_cm'] = target_bottom
        summary['thickness_cm'] = total_thickness
        summary['LONG'] = layer_df['LONG'].iloc[0]
        summary['LAT'] = layer_df['LAT'].iloc[0]
        
        return summary

    def calculate_initial_wsoil(self, theta_fc: float, coarse_fragments: float, layer_depth_mm: float) -> float:
        """
        Calculates the initial soil water content (Wsoil) at Field Capacity.

        Parameters
        ----------
        theta_fc : float
            Field capacity (volumetric fraction or percentage).
        coarse_fragments : float
            Coarse fragments content. 
            NOTE: The current formula uses `coarse_fragments / 10`, which implies 
            units might be on a specific scale (e.g., if input is 10 for 1%). 
            Standard SoilGrids `cfvo` is per mille, which would require `/ 1000`.
        layer_depth_mm : float
            Thickness of the soil layer (mm).

        Returns
        -------
        float
            Soil water content in mm.
        """
        wsoil_mm = theta_fc * layer_depth_mm * (1 - coarse_fragments / 10)
        
        return wsoil_mm

    def get_son(self, layer: int) -> Dict[str, Any]:
        """
        Get the Soil Organic Nitrogen (SON) stock and initial water for a specific layer.

        Parameters
        ----------
        layer : int
            The layer index (e.g., 0 for 0-30 cm, 1 for 30-60 cm).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - 'clay': Weighted average clay content.
            - 'son': Calculated SON stock (kg/ha).
            - 'Wsol': Initial soil water content at field capacity (mm).
        """
        dcm = self.depths[layer][1] - self.depths[layer][0] 
        sp = self.summarize_depths(layer)
        wsol = self.calculate_initial_wsoil(sp['fc'], coarse_fragments=sp['cfvo'], layer_depth_mm=dcm * 10)
        son = self.calculate_son(sp['nitrogen'], dcm, sp['bdod'], sp['cfvo'] / 1000)

        return {
            'clay': sp['clay'],
            'son': son,
            'Wsol': wsol,
        }

    def get_initial_smn(self, weather_manager: Any, starting_date: str, 
                    son_stock: float, clay_pct: float, layer_depth_cm: float, 
                    previous_daysstock: int = 90) -> float:
        """
        Extracts pre-planting weather and calculates the starting Soil Mineral Nitrogen (SMN).

        Uses a 90-day (or specified) window of weather data prior to the starting date
        to estimate the initial mineral N pool.

        Parameters
        ----------
        weather_manager : Any
            Object providing weather data (temperature, precipitation, ET0).

        starting_date : str
            The date to start the simulation (YYYY-MM-DD).
        son_stock : float
            Soil Organic Nitrogen stock (kg/ha).
        clay_pct : float
            Clay content percentage (0-100).
        layer_depth_cm : float
            Thickness of the soil layer (cm).
        previous_daysstock : int, optional
            Number of days of weather data to use for initialization, by default 90.

        Returns
        -------
        float
            The initial soil mineral nitrogen (kg/ha).
        """
        starting_deg = datetime.strptime(starting_date, '%Y-%m-%d') - timedelta(days=previous_daysstock)
        starting_deg_str = datetime.strftime(starting_deg, '%Y-%m-%d')

        tmean_90days = weather_manager.get_mean_temperature(starting_deg_str, starting_date)
        rain_90days = weather_manager.get_precipitation(starting_deg_str, starting_date)
        et0_90days = weather_manager.get_evapotranspiration(starting_deg_str, starting_date)

        bare_soil = np.ones_like(tmean_90days, dtype=bool) 
        
        smn_array = self.calculateSM0(
            son=son_stock, 
            temperature=tmean_90days, 
            rain=rain_90days, 
            evap=et0_90days, 
            clay=clay_pct, 
            soil_depth=layer_depth_cm, 
            bare=bare_soil
        )    
        initial_smn = np.mean(smn_array)
        
        return initial_smn



class BANANASoilMat:
    """
    Tracks and updates soil nitrogen and water parameters for a specific banana mat.

    Attributes
    ----------
    mat_id : int
        Identifier for the mat.
    wsol1 : float
        Water content in soil layer 1 (mm).
    wsol2 : float
        Water content in soil layer 2 (mm).
    SON : float
        Soil organic nitrogen (kg/ha).
    SMN1 : float
        Soil mineral nitrogen in layer 1 (kg/ha).
    SMN2 : float
        Soil mineral nitrogen in layer 2 (kg/ha).
    SMN : float
        Total soil mineral nitrogen (kg/ha).
    """

    def __init__(self, mat_id: int, wsol1: float, wsol2: float, son: float, 
                 smn_depth1: float, smn_depth2: float) -> None:
        """
        Initialize the soil parameters for the mat.

        Parameters
        ----------
        mat_id : int
            The mat identifier.
        wsol1 : float
            Initial water state for layer 1 (mm).
        wsol2 : float
            Initial water state for layer 2 (mm).
        son : float
            Initial soil organic nitrogen (kg/ha).
        smn_depth1 : float
            Initial soil mineral nitrogen at depth 1 (kg/ha).
        smn_depth2 : float
            Initial soil mineral nitrogen at depth 2 (kg/ha).
        """
        self.mat_id = mat_id
        self._initialize()
        
        self.wsol1 = wsol1
        self.wsol2 = wsol2
        self.SON = son
        self.SMN1 = smn_depth1
        self.SMN2 = smn_depth2
        self.SMN = self.SMN1 + self.SMN2
    
    def _initialize(self) -> None:
        """
        Initialize the state variables and physical constants of the soil layers.
        """
        self.dNBAN = 0.0 # N demand of banana plants in mat ‘m’ at time 
        self.pnBAN = 0.0 # Percentage of nitrogen in banana dry biomass in mat
        
        self.dNBAN_1 = 0.0 # Nitrogen demand by soil layer 1
        self.dNBAN_2 = 0.0 # Nitrogen demand by soil layer 2
        
        self.WAL1 = 0.0 # Water available for leaching from the upper soil layer in mat ‘m’ at time step ‘t’
        self.WAL = 0.0 # Water available for leaching from the lower soil layer in mat 
        
        self.ET1 = 0.0 # soil 1
        self.ET2 = 0.0 # soil 2
        
        # nitrogen balance
        self.NAL1 = 0.0
        self.NAL2 = 0.0
        self.NAL = 0.0 
        self.MOS = 0.0 # Mineral N from mineralization of soil organic matter in mat ‘m’ at time step ‘t’
        
        # water storage
        self.SW1 = 200.0 
        self.SW2 = 220.0
        
        
        self.kl1 = 0.7 # leaching coefficient upper layer
        self.kl2 = 0.6 # leaching coefficient lower layer
        
        ##
        self.UBAN1 = 0.0 #N uptake of banana plants in the upper soil layer in mat ‘m’ at time step ‘t’
        self.UBAN2 = 0.0 #N uptake of banana plants in the upper soil layer in mat ‘m’ at time step ‘t’

