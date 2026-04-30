"""
ag_cube_cm.models.dssat.base
=============================

DSSAT CropModel Implementation.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xarray as xr
from pyproj import Transformer

from ag_cube_cm.models.base import CropModel
from ag_cube_cm.models.factory import register_model

logger = logging.getLogger(__name__)


@register_model("dssat")
class DSSATModel(CropModel):
    """
    DSSAT Implementation of the CropModel ABC.
    
    This model orchestrates the writing of strict fixed-width Fortran files
    (.WTH, .SOL, .X), dynamically generates a batch file (DSSBatch.v48),
    executes the binary via subprocess, and parses Summary.OUT.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.crop = getattr(self.config.CROP, 'name', 'Maize').lower()
        self.cultivar = getattr(self.config.CROP, 'cultivar', None)
        self.crop_code = self._get_crop_code(self.crop)
        self.exp_filename = None
        self._soil_layers: list = []  # [(depth_cm, slll, sdul), ...]
        self._nyers: int = 1          # auto-detected from weather span in prepare_inputs

    def _get_crop_code(self, crop: str) -> str:
        """Map crop names to DSSAT 2-letter codes."""
        codes = {
            "maize": "MZ", "millet": "ML", "sugarbeet": "BS", "rice": "RI",
            "sorghum": "SG", "sweetcorn": "SW", "alfalfa": "AL", "bermudagrass": "BM",
            "soybean": "SB", "canola": "CN", "sunflower": "SU", "potato": "PT",
            "tomato": "TM", "cabbage": "CB", "sugarcane": "SC", "wheat": "WH",
            "bean": "BN", "cassava": "CS"
        }
        return codes.get(crop.lower(), "MZ")

    def prepare_inputs(self, weather_slice: xr.Dataset, soil_slice: xr.Dataset,
                       elevation: float = -99.0) -> None:
        """
        Convert Xarray slices into strict DSSAT Fortran formats inside the
        isolated working directory.

        Parameters
        ----------
        elevation : float
            Site elevation in metres (written to .WTH header).  Pass -99 when
            no DEM is available.
        """
        if not self.working_dir:
            raise RuntimeError("Working directory not setup. Call setup_working_directory() first.")

        df_wth = weather_slice.to_dataframe().reset_index().dropna()
        df_sol = soil_slice.to_dataframe().reset_index().dropna()

        # Extract coordinates for header files — reproject to WGS84 if needed
        raw_y = float(df_sol['y'].iloc[0]) if 'y' in df_sol.columns else 0.0
        raw_x = float(df_sol['x'].iloc[0]) if 'x' in df_sol.columns else 0.0
        src_crs = None
        try:
            src_crs = soil_slice.rio.crs
        except Exception:
            pass
        if src_crs and src_crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(raw_x, raw_y)
        else:
            lat, lon = raw_y, raw_x

        # Unit conversions for AgERA5 data
        # Temperatures: Kelvin → Celsius
        for temp_col in ['tmin', 'tmax']:
            if temp_col in df_wth.columns and df_wth[temp_col].mean() > 100:
                df_wth[temp_col] = df_wth[temp_col] - 273.15
        # Solar radiation: J m-2 d-1 → MJ m-2 d-1
        if 'srad' in df_wth.columns and df_wth['srad'].mean() > 10000:
            df_wth['srad'] = df_wth['srad'] / 1e6

        if not df_wth.empty:
            self._write_wth(df_wth, lat, lon, elevation)
            # Auto-detect NYERS: seasons that fit between planting year and last weather year.
            # Config can override via number_years; 0 or None triggers auto-detection.
            cfg_nyers = getattr(self.config.GENERAL_INFO, 'number_years', None)
            if cfg_nyers:
                self._nyers = int(cfg_nyers)
            else:
                planting_date_val = getattr(self.config.MANAGEMENT, 'planting_date', None)
                pdate_dt = datetime.strptime(
                    str(planting_date_val) if planting_date_val else '2000-01-01', '%Y-%m-%d'
                )
                planting_year = pdate_dt.year
                last_wth_year = int(pd.to_datetime(df_wth['date']).dt.year.max())
                # If harvest (~200 days after planting) crosses into the next calendar year,
                # the last simulated season would need weather past last_wth_year — drop it.
                harvest_crosses_year = (pdate_dt + timedelta(days=200)).year > pdate_dt.year
                nyers_raw = last_wth_year - planting_year + 1
                self._nyers = max(1, nyers_raw - (1 if harvest_crosses_year else 0))

        if not df_sol.empty:
            self._write_sol(df_sol, lat, lon)

        self._write_mzx(lat, lon)

    def _write_wth(self, df_wth: pd.DataFrame, lat: float, lon: float,
                   elevation: float = -99.0) -> None:
        """
        Write the .WTH file with strict Fortran 0-padded spacing.
        Maintains the exact column spacing required by DSSAT.
        """
        file_path = self.working_dir / "WTHE0001.WTH"

        # DSSAT requires strictly chronological rows — sort by date first.
        df_wth = df_wth.copy()
        df_wth['_date_parsed'] = pd.to_datetime(df_wth['date'])
        df_wth = df_wth.sort_values('_date_parsed').drop(columns=['_date_parsed'])

        # Calculate Long-Term Average Temp (TAV) and Amplitude (AMP)
        if 'tmax' in df_wth.columns and 'tmin' in df_wth.columns:
            tav = ((df_wth['tmax'] + df_wth['tmin']) / 2).mean()

            # Simple AMP approximation
            df_wth = df_wth.copy()
            df_wth['month'] = pd.to_datetime(df_wth['date']).dt.month
            monthly_mean = ((df_wth['tmax'] + df_wth['tmin']) / 2).groupby(df_wth['month']).mean()
            amp = (monthly_mean.max() - monthly_mean.min()) / 2
        else:
            tav = 20.0
            amp = 5.0

        elev_str = f"{int(elevation):5d}" if elevation != -99.0 else "  -99"

        with open(file_path, "w") as f:
            f.write("*WEATHER DATA : EXPS\n\n")
            f.write("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT\n")
            f.write(f"  EXPS {lat:8.3f} {lon:8.3f} {elev_str} {tav:5.1f} {amp:5.1f}   -99   -99\n")
            f.write("@DATE  SRAD  TMAX  TMIN  RAIN\n")
            
            for _, row in df_wth.iterrows():
                # Date format: YYJJJ (e.g., 2001-01-01 -> 01001)
                yyjjj = pd.to_datetime(row['date']).strftime("%y%j")
                
                srad = row.get('solar_radiation', row.get('srad', -99.0))
                tmax = row.get('tmax', -99.0)
                tmin = row.get('tmin', -99.0)
                rain = row.get('precipitation', row.get('rain', -99.0))
                
                # Write row with strict Fortran spacing.
                # @DATE(5) + Space(1) + SRAD(5) + Space(1) + TMAX(5)...
                f.write(f"{yyjjj:>5} {srad:5.1f} {tmax:5.1f} {tmin:5.1f} {rain:5.1f}\n")

    def _write_sol(self, df_sol: pd.DataFrame, lat: float, lon: float) -> None:
        """
        Write the .SOL file using standard variables.
        """
        file_path = self.working_dir / "TR.SOL"
        country = getattr(self.config.GENERAL_INFO, 'country', 'USA')
        
        # Default bulk averages for top properties
        salb, slu1, sldr = 0.13, 1.0, 0.25 
        
        with open(file_path, "w") as f:
            f.write("*SOILS: EXPS\n\n")
            f.write("*TRAN0001  UNCLASSIFIED             SCL       100\n")
            f.write("@SITE        COUNTRY          LAT     LONG SCS FAMILY\n")
            f.write(f" EXPS        {country[:10]:<10}  {lat:8.3f} {lon:8.3f}\n")
            f.write("@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n")
            f.write(f"   -99 {salb:5.2f} {slu1:5.1f} {sldr:5.2f}  73.0  1.00  1.00 -99.0 -99.0 -99.0\n")
            f.write("@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n")
            
            # Write layers, storing (depth, slll, sdul) for use in IC section of MZX
            self._soil_layers = []
            for _, row in df_sol.iterrows():
                depth = row.get('depth', -99)
                if isinstance(depth, str) and '-' in depth:
                    depth = int(depth.split('-')[1])
                elif isinstance(depth, str):
                    depth = int(depth)
                else:
                    depth = int(depth)

                # SoilGrids v2 scale factors → DSSAT units
                slll = row.get('wv1500', -99.0) / 1000  # 10^-3 cm³/cm³ → cm³/cm³
                sdul = row.get('wv0033', -99.0) / 1000
                ssat = row.get('wv0010', -99.0) / 1000
                sbdm = row.get('bdod',   -99.0) / 100   # cg/cm³ → g/cm³
                sloc = row.get('soc',    -99.0) / 100   # dg/kg → %
                slcl = row.get('clay',   -99.0) / 10    # g/kg → %
                slsi = row.get('silt',   -99.0) / 10    # g/kg → %
                slcf = row.get('cfvo',   -99.0) / 10    # cm³/dm³ × 10 → %
                slhw = row.get('phh2o',  -99.0) / 10    # pH × 10 → pH

                self._soil_layers.append((depth, slll, sdul))
                f.write(f"{depth:6d}   -99 {slll:5.3f} {sdul:5.3f} {ssat:5.3f} 1.000 -99.0 {sbdm:5.2f} {sloc:5.2f} {slcl:5.1f} {slsi:5.1f} {slcf:5.1f} -99.0 -99.0 {slhw:5.2f} -99.0 -99.0 -99.0\n")

    def _write_mzx(self, lat: float, lon: float) -> None:
        """
        Generate the experiment (.MZX) file matching the legacy KEAG8104 template format.
        """
        self.exp_filename = f"EXPS0001.{self.crop_code}X"
        file_path = self.working_dir / self.exp_filename

        planting_date_val = getattr(self.config.MANAGEMENT, 'planting_date', None)
        planting_date = str(planting_date_val) if planting_date_val is not None else '2000-01-01'
        pdate = datetime.strptime(planting_date, '%Y-%m-%d')
        pdate_str = pdate.strftime("%y%j")
        plname = pdate.strftime("%d-%b")

        # Simulation start = 1 month before planting
        sdate = (pdate - timedelta(days=30)).strftime("%y%j")
        # Harvest reference date = planting + 200 days (HARVS=R means DSSAT harvests at maturity)
        hdate = (pdate + timedelta(days=200)).strftime("%y%j")
        # Planting window ± 3 days for automatic management
        pfrst = (pdate - timedelta(days=3)).strftime("%y%j")
        plast = (pdate + timedelta(days=3)).strftime("%y%j")

        fert_apps = getattr(self.config.MANAGEMENT, 'fertilizer_schedule', [])
        has_fert = bool(fert_apps)
        mf_flag = 1 if has_fert else 0
        nyers = self._nyers  # set by prepare_inputs from weather-data span

        crop_module = self._CROPS_MODULES.get(self.crop, "MZCER")

        with open(file_path, "w") as f:
            f.write(f"*EXP.DETAILS: EXPS0001{self.crop_code} SIMULATION\n\n")

            f.write("*GENERAL\n")
            f.write("@PEOPLE\nCGIAR\n@ADDRESS\nCGIAR\n@SITE\nCGIAR\n")
            f.write("@ PAREA  PRNO  PLEN  PLDR  PLSP  PLAY HAREA  HRNO  HLEN  HARM.........\n")
            f.write("    450     4    30   -99   -99   -99   450     4   -99   -99\n\n")

            f.write("*TREATMENTS                        -------------FACTOR LEVELS------------\n")
            f.write("@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n")
            f.write(f" 1 1 1 0 Default                    1  1  0  1  1  0  {mf_flag}  0  0  0  0  1  1\n\n")

            f.write("*CULTIVARS\n")
            f.write("@C CR INGENO CNAME\n")
            f.write(f" 1 {self.crop_code} {self.cultivar if self.cultivar else '999991'} NONE\n\n")

            f.write("*FIELDS\n")
            f.write("@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME\n")
            f.write(" 1 UFWA0001 WTHE0001   -99   -99   -99   -99   -99   -99   SL   -99  TRAN0001   -99\n")
            f.write("@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR\n")
            f.write(f" 1 {lon:15.3f} {lat:15.3f}       -99               -99   -99   -99   -99 FH301     0\n\n")

            f.write("*INITIAL CONDITIONS\n")
            f.write("@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME\n")
            f.write(f" 1    {self.crop_code} {sdate}   150   -99  1.00  1.00   -99  1512  1.10  0.10     0   -99 -99\n")
            f.write("@C  ICBL  SH2O  SNH4  SNO3\n")
            # Derive SH2O from actual soil layers: slll + 80% of plant-available water
            if self._soil_layers:
                for (depth, slll, sdul) in self._soil_layers:
                    sh2o = slll + 0.8 * max(sdul - slll, 0.0)
                    sh2o = max(sh2o, slll + 0.05)  # ensure above wilting point
                    f.write(f" 1 {depth:5d} {sh2o:5.3f}  0.60  3.00\n")
            else:
                f.write(" 1    20 0.250  0.60  3.00\n")
                f.write(" 1    60 0.250  0.60  3.00\n")
            f.write("\n")

            f.write("*PLANTING DETAILS\n")
            f.write("@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME\n")
            f.write(f" 1 {pdate_str}   -99   5.0   5.0     S     R    75     0     5   -99   -99   -99   -99     0                        {plname}\n\n")

            f.write("*FERTILIZERS (INORGANIC)\n")
            f.write("@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME\n")
            if has_fert:
                for fert in fert_apps:
                    dap   = getattr(fert, 'days_after_planting', 0)
                    fdate = (pdate + timedelta(days=dap)).strftime("%y%j")
                    n_amt = getattr(fert, 'n_kg_ha', 0.0)
                    p_amt = getattr(fert, 'p_kg_ha', 0.0)
                    k_amt = getattr(fert, 'k_kg_ha', 0.0)
                    f.write(f" 1 {fdate} FE001 AP001     5 {n_amt:5.1f} {p_amt:5.1f} {k_amt:5.1f}   -99   -99   -99 Application\n")
            else:
                f.write(f" 1 {pdate_str} FE006   -99     4   -99   -99   -99   -99   -99   -99 Nofertilizer\n")
            f.write("\n")

            f.write("*HARVEST DETAILS\n")
            f.write("@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC HNAME\n")
            f.write(f" 1 {hdate} GS006     H     A   100     0 {self.crop.upper()[:5]}\n\n")

            f.write("*SIMULATION CONTROLS\n")
            f.write("@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL\n")
            f.write(f" 1 GE             {nyers:2d}     1     S {sdate}  2150 DEFAULT                  {crop_module}\n")
            f.write("@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2\n")
            f.write(" 1 OP              Y     Y     Y     N     N     N     N     Y     M\n")
            f.write("@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL\n")
            f.write(" 1 ME              M     M     E     R     S     C     R     1     G     S     2\n")
            f.write("@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS\n")
            ferti_flag = "R" if has_fert else "N"
            f.write(f" 1 MA              R     N     {ferti_flag}     N     R\n")
            f.write("@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT\n")
            f.write(" 1 OU              N     N     Y     1     N     N     N     N     N     N     0     N     N     A\n\n")

            f.write("@  AUTOMATIC MANAGEMENT\n")
            f.write("@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n")
            f.write(f" 1 PL          {pfrst} {plast}    40   100    30    40    10\n")
            f.write("@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n")
            f.write(" 1 IR              0     0     0 GS000 IR001     0     1\n")
            f.write("@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n")
            f.write(" 1 NI             30    50    25 FE001 GS000\n")
            f.write("@N RESIDUES    RIPCN RTIME RIDEP\n")
            f.write(" 1 RE            100     1    20\n")
            f.write("@N HARVEST     HFRST HLAST HPCNP HPCNR\n")
            f.write(" 1 HA             -99   -99   100     0\n")

    def _generate_batch_file(self) -> None:
        """Kept for reference; not used — direct C-mode execution avoids path-space issues."""
        pass

    # ------------------------------------------------------------------
    # Static folder bundled with this package (contains binaries,
    # Genotype/, StandardData/, Soil/, *.CDE files)
    # ------------------------------------------------------------------
    _STATIC_PATH: Path = Path(__file__).parent / "static"
    _DSSAT_VERSION: str = "048"
    _bootstrap_lock: threading.Lock = threading.Lock()  # one bootstrap at a time

    # DSSAT module name by crop (used in DSSATPRO config file M-line)
    _CROPS_MODULES: dict = {
        "maize":        "MZCER",
        "millet":       "MLCER",
        "rice":         "RICER",
        "sorghum":      "SGCER",
        "wheat":        "WHCER",
        "sweetcorn":    "MZCER",
        "soybean":      "CRGRO",
        "bean":         "CRGRO",
        "tomato":       "CRGRO",
        "cabbage":      "CRGRO",
        "alfalfa":      "CRGRO",
        "bermudagrass": "CRGRO",
        "sugarbeet":    "BSCER",
        "sugarcane":    "SCCAN",
        "potato":       "PTSUB",
        "cassava":      "CSYCA",
        "sunflower":    "SUOIL",
        "canola":       "CNOIL",
    }

    def _copy_genotype_files(self) -> None:
        """Copy crop-specific .CUL / .ECO / .SPE files from static/Genotype/
        into self.working_dir so DSSAT can find them without relying on CRD path
        resolution (which varies across DSSAT binary versions).
        """
        crop_module = (
            "CRGRO"
            if self.crop_code == "BN"
            else self._CROPS_MODULES.get(self.crop, "MZCER")
        )
        genotype_dir = self._STATIC_PATH / "Genotype"
        for ext in ("CUL", "ECO", "SPE"):
            src = genotype_dir / f"{crop_module}{self._DSSAT_VERSION}.{ext}"
            if src.exists():
                shutil.copy2(str(src), str(self.working_dir / src.name))
            else:
                logger.warning("Genotype file not found: %s", src)

    def _write_confile(self, dssat_home: Path) -> None:
        """Write DSSATPRO.V48 (Windows) or DSSATPRO.L48 (Linux/Mac) to
        self.working_dir.

        Mirrors DSSATTools create_dssat_config_path_file:
            WED    <weather/workdir>
            M{crop_code}    <workdir> dscsm048 {module}{version}
            CRD    <Genotype/>
            PSD    <Pest/>
            SLD    <Soil/>
            STD    <StandardData/>
        """
        is_windows = platform.system().lower() == "windows"
        confile = "DSSATPRO.V48" if is_windows else "DSSATPRO.L48"

        # Resolve module name (CRGRO for legumes, otherwise from map)
        crop_module = (
            "CRGRO"
            if self.crop_code == "BN"
            else self._CROPS_MODULES.get(self.crop, "MZCER")
        )

        def _abs(path: Path) -> str:
            """Absolute path string, no trailing separator."""
            return str(path.resolve())

        workdir_abs = _abs(self.working_dir)

        # DSSAT DSSATPRO.V48 uses whitespace-delimited parsing for every line.
        # A space anywhere in the path corrupts the M-line and produces rc=99
        # ("Crop code incompatible with model specified"). Fail fast with a clear message.
        if ' ' in workdir_abs:
            raise RuntimeError(
                f"DSSAT working directory path contains spaces, which DSSAT cannot handle "
                f"(DSSATPRO whitespace-delimiter bug):\n  {workdir_abs}\n"
                f"Set GENERAL_INFO.working_path to a path without spaces, "
                f"e.g. 'D:/tmp/dssat_runs' or '/tmp/dssat_runs'."
            )

        # WED must point to the weather FILE (stem only, no extension or trailing slash).
        # WTH file is always named WTHE0001.WTH — DSSAT appends .WTH itself.
        wth_stem = workdir_abs + os.sep + "WTHE0001"

        with open(self.working_dir / confile, "w") as f:
            f.write(f"WED    {wth_stem}\n")
            # MMZ path = run directory WITHOUT trailing separator; DSSAT appends sep+filename
            f.write(f"M{self.crop_code}    {workdir_abs} dscsm048 "
                    f"{crop_module}{self._DSSAT_VERSION}\n")
            f.write(f"CRD    {_abs(dssat_home / 'Genotype')}\n")
            f.write(f"PSD    {_abs(dssat_home / 'Pest')}\n")
            f.write(f"SLD    {_abs(dssat_home / 'Soil')}\n")
            f.write(f"STD    {_abs(dssat_home / 'StandardData')}\n")

    def _bootstrap_dssat_home(self, tmp_base: str) -> tuple:
        """Copy the bundled static/ folder to <tmp_base>/DSSAT048/ and return
        (dssat_home: Path, bin_path: Path).  Skips files that already exist so
        repeated calls in the same session are cheap.
        """
        dssat_home = Path(tmp_base) / f"DSSAT{self._DSSAT_VERSION}"

        with self._bootstrap_lock:
            dssat_home.mkdir(parents=True, exist_ok=True)
            for item in self._STATIC_PATH.iterdir():
                dest = dssat_home / item.name
                if item.is_dir():
                    if not dest.exists():
                        shutil.copytree(str(item), str(dest))
                else:
                    try:
                        if dest.exists():
                            dest.unlink()
                        shutil.copy2(str(item), str(dest))
                    except OSError:
                        pass  # another thread already wrote it; skip

        is_windows = platform.system().lower() == "windows"
        exe_name = "dscsm048.exe" if is_windows else "dscsm048"
        bin_path = dssat_home / "bin" / exe_name

        if not bin_path.exists():
            raise FileNotFoundError(
                f"DSSAT binary not found in bundled static folder: {bin_path}"
            )

        # Ensure the binary is executable on Linux/Mac
        if not is_windows:
            bin_path.chmod(bin_path.stat().st_mode | 0o111)

        return dssat_home, bin_path

    def run_simulation(self) -> None:
        """Execute the DSSAT binary via subprocess.

        Two execution paths:
        1. Config-provided — ``bin_path`` exists on disk: use it directly with
           ``dssat_path`` as DSSAT_HOME.
        2. Bundled bootstrap — no ``bin_path`` (or path is missing): copy
           ``static/`` to ``<dssat_path or tempdir>/DSSAT048/`` and run from
           there (follows the DSSATTools pattern).
        """
        cfg_bin   = getattr(self.config.GENERAL_INFO, 'bin_path',   None)
        cfg_home  = getattr(self.config.GENERAL_INFO, 'dssat_path', None)

        if cfg_bin and os.path.exists(cfg_bin):
            # Path 1 — user-supplied binary
            bin_path   = cfg_bin
            dssat_home = Path(cfg_home or os.path.dirname(cfg_bin))
            logger.debug("[Pixel %s] Using config-supplied DSSAT binary: %s", self.pixel_id, bin_path)
        else:
            # Path 2 — bootstrap from bundled static/
            tmp_base = cfg_home or tempfile.gettempdir()
            logger.debug("[Pixel %s] Bootstrapping DSSAT from static/ into %s", self.pixel_id, tmp_base)
            dssat_home, bin_path_obj = self._bootstrap_dssat_home(tmp_base)
            bin_path = str(bin_path_obj)

        # Copy crop .CUL/.ECO/.SPE into run dir (DSSAT needs them locally).
        self._copy_genotype_files()
        # Write DSSATPRO.V48/.L48 to the pixel working directory.
        self._write_confile(dssat_home)

        # Use C mode (direct execution): dscsm048 C EXPS0001.MZX 1
        # This avoids the DSSBatch.v48 71-char padded-path bug in DSSAT 4.8.
        cmd = [bin_path, "C", self.exp_filename, "1"]

        try:
            # Inherit the full parent environment so PATH / LD_LIBRARY_PATH / HOME
            # are available on Linux; only override DSSAT_HOME.
            run_env = os.environ.copy()
            run_env["DSSAT_HOME"] = str(dssat_home)

            result = subprocess.run(
                cmd,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                env=run_env,
            )
            logger.debug("[Pixel %s] DSSAT stdout: %s", self.pixel_id, result.stdout[-500:])
            summary = self.working_dir / "Summary.OUT"
            if not summary.exists():
                # Crop failure (rc=3) or other non-fatal DSSAT result —
                # collect_outputs will return {} and the caller treats it as NaN yield.
                last_lines = result.stdout.strip().splitlines()
                diag = " | ".join(last_lines[-3:]) if last_lines else "no output"
                logger.warning(
                    "[Pixel %s] DSSAT finished without Summary.OUT (rc=%d): %s",
                    self.pixel_id, result.returncode, diag,
                )
            else:
                logger.debug("[Pixel %s] DSSAT executed successfully.", self.pixel_id)

        except subprocess.CalledProcessError as e:
            logger.error("[Pixel %s] DSSAT failed (rc=%d)", self.pixel_id, e.returncode)
            logger.error("[Pixel %s] STDOUT: %s", self.pixel_id, e.stdout)
            logger.error("[Pixel %s] STDERR: %s", self.pixel_id, e.stderr)
            raise RuntimeError(f"DSSAT execution failed for pixel {self.pixel_id}")

    def collect_outputs(self) -> Dict[str, Any]:
        """
        Parse Summary.OUT and extract relevant yield data into a dictionary.
        """
        summary_file = self.working_dir / "Summary.OUT"
        if not summary_file.exists():
            return {}
            
        result_dict = {}
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            
        # Locate the header line and all data rows (one per simulated year)
        header_line = None
        data_rows = []
        for i, line in enumerate(lines):
            if "@RUNNO" in line or ("@" in line and "HWAM" in line):
                header_line = line
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith('!') and not stripped.startswith('@'):
                        data_rows.append(lines[j])
                break

        if not header_line or not data_rows:
            return result_dict

        headers = header_line.replace('@', '').split()

        # Parse every row (one per NYERS season)
        all_years: list[dict] = []
        for row in data_rows:
            row_dict: dict = {}
            values = row.split()
            for h, v in zip(headers, values):
                try:
                    row_dict[h] = float(v)
                except ValueError:
                    row_dict[h] = v
            all_years.append(row_dict)

        # result_dict uses first-year values for metadata fields (PDAT, MDAT, …)
        result_dict.update(all_years[0])

        if len(all_years) == 1:
            return result_dict

        # Multi-year: add per-year HWAM list and overwrite scalar HWAM with mean
        hwam_values = [
            r["HWAM"] for r in all_years
            if isinstance(r.get("HWAM"), float) and r["HWAM"] > 0
        ]
        result_dict["HWAM_yearly"] = [r.get("HWAM", float("nan")) for r in all_years]
        result_dict["HWAM"] = (
            sum(hwam_values) / len(hwam_values) if hwam_values else float("nan")
        )
        return result_dict
