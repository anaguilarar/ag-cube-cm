from datetime import datetime
from typing import Generator, Dict

import numpy as np
import pandas as pd

class Weather():
    """
    A class to manage and process meteorological data for a specific station.

    Parameters
    ----------
    path : str, optional
        The file path to a CSV containing weather data.
    df : pd.DataFrame, optional
        A pandas DataFrame containing weather data.
    station : Any, optional
        A station object that handles day entry and ET calculations.

    Attributes
    ----------
    weather : pd.DataFrame or None
        The underlying weather dataset.
    station : Any or None
        The station associated with this weather data.
    wind_speed : float or None
        Current day's wind speed.
    tmax : float or None
        Current day's maximum temperature.
    tmin : float or None
        Current day's minimum temperature.
    srad : float or None
        Current day's solar radiation.
    rain : float or None
        Current day's rainfall.
    vapour_pressure : float or None
        Current day's vapour pressure.
    julian_day : int or None
        The day of the year.
    """
    
    def _initialize(self):
        self.wind_speed = None
        self.tmax = None
        self.tmin = None
        self.srad = None
        self.rain = None
        self.vapour_pressure = None
        self.julian_day = None
    
    @property
    def weather(self):
        if self._weather is None and self.path is not None:
            self._weather = pd.read_csv(self.path)
        
        return self._weather
    
    
    def __init__(self, path:str = None, df:pd.DataFrame = None):

        self._weather = None
        self.path = path
        if df is not None:
            self._weather = df   
        


class BanWeather(Weather):
    """
    Weather data handler specifically for the Banana Nitrogen model.

    Inherits from the base Weather class to provide specialized filtering 
    and ETo calculation for banana crop cycles.

    Attributes
    ----------
    variables_dict : dict
        Mapping of model variable names to dataset column names.
    """

    @property
    def variables_dict(self):
        """dict: Internal mapping of weather variables."""
        return {
            "year": "year",
            "precipitation": "rain",
            "tmin": "tmin",
            "tmax": "tmax",
            "vp": "vapour_pressure",
            "wind_speed": "wind_speed",
            "solar_radiation": "srad",
            "etr": "etr"
        }


    def __init__(self, path=None, df=None, latitude:float=None, longitude:float=None, altitude:int=None , date_format="%Y%m%d"):        
        """
        Initialize the weather station and load data.

        Parameters
        ----------
        path : str, optional
            Path to the weather data file.
        df : pd.DataFrame, optional
            DataFrame containing weather records.
        latitude : float, optional
            Station latitude in decimal degrees.
        longitude : float, optional
            Station longitude in decimal degrees.
        altitude : float, optional
            Station altitude in meters.
        date_format : str, optional
            data frame date format, default '%Y%m%d'
        """

        super().__init__(path, df)
        self.weather.DATE = pd.to_datetime(self.weather.DATE , format=date_format)
        self._weather_subset: pd.DataFrame = None
        self.starting_date: datetime

    def _filter_weather_by_date(self, starting_date: str, ending_date: str):
        """
        Subset the weather data for a specific date range.

        Parameters
        ----------
        starting_date : str or datetime
            Start of the simulation period (YYYY-MM-DD).
        ending_date : str or datetime
            End of the simulation period (YYYY-MM-DD).
        """
        if isinstance(starting_date, str):
            starting_date = datetime.strptime(starting_date, '%Y-%m-%d')
        if isinstance(ending_date, str):
            ending_date = datetime.strptime(ending_date, '%Y-%m-%d')

        self._weather_subset = self.weather.loc[np.logical_and(self.weather.DATE>=starting_date, self.weather.DATE<=ending_date)]

    def get_precipitation(self, starting_date, ending_date):
        """
        Get precipitation values for a specific period.

        Parameters
        ----------
        starting_date : str
            Start date.
        ending_date : str
            End date.

        Returns
        -------
        numpy.ndarray
            Array of daily precipitation (mm).
        """
        self._filter_weather_by_date(starting_date, ending_date)
        return self._weather_subset[self.variables_dict["precipitation"]].values

    def get_degree_thermal_time(self, starting_date: str, ending_date: str, tbase = 14.7):
        self._filter_weather_by_date(starting_date, ending_date)
        df_copy = self._weather_subset.copy()

        df_copy['daily_tmean'] = (df_copy['tmax'] + df_copy['tmin']) / 2
        dtt = (df_copy['daily_tmean'] - tbase).values.copy()
        dtt[dtt<0] = 0
        return dtt

    def get_mean_temperature(self, starting_date, ending_date):
        """
        Get daily mean temperature for a specific period.

        Calculated as (Tmin + Tmax) / 2.

        Parameters
        ----------
        starting_date : str
            Start date.
        ending_date : str
            End date.

        Returns
        -------
        numpy.ndarray
            Array of daily mean temperatures (°C).
        """
        self._filter_weather_by_date(starting_date, ending_date)
        tmin_values = self._weather_subset[self.variables_dict["tmin"]].values
        tmax_values = self._weather_subset[self.variables_dict["tmax"]].values
        return (tmin_values + tmax_values) / 2

    def get_evapotranspiration(self, starting_date, ending_date, eto_method: str=None):
        """
        Calculate daily reference evapotranspiration (ETo) for a specific period.

        Parameters
        ----------
        starting_date : str
            Start date.
        ending_date : str
            End date.
        eto_method : {'PM', 'PT'}, optional
            Method for ETo calculation:
            - 'PM': Penman-Monteith (standard).
            - 'PT': Priestley-Taylor.
            Defaults to 'PM'.

        Returns
        -------
        numpy.ndarray
            Array of daily ETo values (mm).
        """
        self._filter_weather_by_date(starting_date, ending_date)

        days = self._weather_subset.index.values
        et0 = np.zeros(self._weather_subset.shape[0])
        if "etr" not in self.weather.columns:            
            for i, day in enumerate(days):
                self.get_day_weather(day)
                et0[i] = self.get_day_eto(eto_method = eto_method)

            return et0
        else:
            return self._weather_subset.etr.values
        
        
    def weekly_weather(self, starting_date, ending_date, add_variable:Dict = None, tbase = 14.7) -> pd.DataFrame:
        
        dtt = self.get_degree_thermal_time(starting_date, ending_date, tbase)
        self._filter_weather_by_date(starting_date, ending_date)

        df_copy = self._weather_subset.copy()
        df_copy.set_index("DATE", inplace=True)
        df_copy['dtt'] = dtt
        # Resample to weekly frequency
        summary_parameters = {
            "srad": "sum",
            "tmin": "sum",
            "tmax": "sum",
            "rain": "sum",
            "dtt": "sum"
        }
        if "etr" in df_copy.columns:
            summary_parameters["etr"] = "sum"
            
        if add_variable: summary_parameters.update(add_variable)

        weekly_df = df_copy.resample("7D", origin=starting_date).agg(summary_parameters).reset_index()
        
        weekly_df = weekly_df[weekly_df['DATE'] >= pd.to_datetime(starting_date)]
        
        weekly_df["year"] = weekly_df["DATE"].dt.year
        weekly_df["doy"] = weekly_df["DATE"].dt.dayofyear
        weekly_df["DATE"] = weekly_df["DATE"].dt.strftime("%Y%m%d")

        # Ensure TMAX is not less than TMIN
        if not all(weekly_df.tmin <= weekly_df.tmax):
            weekly_df.loc[(weekly_df.tmin > weekly_df.tmax),"tmax"] = weekly_df.loc[(weekly_df.tmin > weekly_df.tmax),"tmin"]+1

        weekly_df["tmean"] = (weekly_df["tmax"] + weekly_df["tmin"])/2

        return weekly_df