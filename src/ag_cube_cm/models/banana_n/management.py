from typing import Dict, Sequence

import numpy as np

def nitrogen_release(Cr0: float, r: float, Y: float, L: float, t: float, h: float, wr: float, wb: float, wh: float = 0.1) -> Dict[str, float]:
    """
    STICS crop model equation for the carbon in the residue pool over time.
    Calculates carbon pool states and fluxes based on the initial carbon, 
    the decomposition rate, assimilation yield, microbial death rate, and time.
    
    Parameters
    ----------
    Cr0 : float
        The Initial Carbon dumped on the field (e.g., from chopped banana leaves or fertilizer).
    r : float
        The Decomposition Rate of the raw material.
    Y : float
        The Assimilation Yield. Fraction assimilated by microbes into their bodies.
    L : float
        The Microbial Death Rate.
    t : float
        The Time (in weeks) since the harvest happened or the fertilizer was applied.
    h : float
        The Humification Rate. Fraction of decomposing carbon converted into humus.
    wr : float
        The N:C ratio of the raw material.
    wb : float
        The N:C ratio of the microbial biomass.
    wh : float, optional
        The N:C ratio of the humus. Default is 0.1.

    Returns
    -------
    Dict[str, float]
        A dictionary containing state variables and fluxes for carbon and nitrogen:
        - cb, ch, cr: Carbon in biomassa, humus and raw residue
        - dCb, dChum, dCr: Delta of carbon in biomass, humus and raw residue
        - dNb, dNhum, dNr: Delta of nitrogen in biomass, humus and raw residue
        - dNres: Mineral nitrogen released.
    """
    cr = Cr0 * np.exp(-r * t)
    cb = ((r * Y * Cr0) / (L - r)) * (np.exp(-r * t) - np.exp(-L * t))
    ch = Cr0 * ((Y * h / (L - r)) * (r * np.exp(-L * t) - L * np.exp(-r * t)) + Y * h)
    
    dCh = L * h * cb # Humus formed this week.
    dCr = -r * cr # How much raw residue rots this week. (It is negative because the residue tank is losing mass).
    dCb = r * Y * cr - L * cb  # The change in microbe population this week.
    
    dNr = dCr * wr # Nitrogen leaving the raw residue this week
    dNb = dCb * wb # Nitrogen absorbed or released by microbes this week
    dNhum = dCh * wh # Nitrogen locked into new humus this week
    dNres = - dNr - dNb - dNhum # Mineral nitrogen released into the soil this week from the residue decomposition
    
    return {
        "cr": cr, "cb": cb, "ch": ch,
        "dCr": dCr, "dCb": dCb, "dChum": dCh,
        "dNr": dNr, "dNb": dNb, "dNhum": dNhum,
        "dNres": dNres
    }

class BANANAFerti:
    """
    Manages organic fertilizer application and its subsequent mineralization.

    Attributes
    ----------
    of_type : str
        Type of organic fertilizer ('Abflor', 'compost', 'bagasse', 'Fertisol', 'Vegegwa').
    SOMap : float
        Weeks since application.
    Cr0OF : float
        Initial carbon from organic fertilizer.
    CrOF : float
        Current raw carbon pool from fertilizer.
    CbOF : float
        Carbon in microbial biomass pool from fertilizer.
    ChOF : float
        Carbon in humus pool from fertilizer.
    dNhumOF : float
        Nitrogen humidified this week from fertilizer.
    dNrOF : float
        Nitrogen released from raw fertilizer this week.
    dNbOF : float
        Nitrogen absorbed/released by microbes from fertilizer.
    dNRESOF : float
        Net mineral nitrogen released to soil this week from fertilizer.
    """
    def organic_parameters(self) -> None:
        """Loads specific parameters for the selected organic fertilizer type."""
        o_parameters = {
            "Abflor": {
                "CNROF": 4.0,           # C:N ratio of the fertilizer [cite: 3156]
                "CNBOF": 7.8,           # C:N ratio of zymogenous microbial biomass [cite: 3156]
                "pcORG": 0.3314,        # Fraction of carbon in the fertilizer [cite: 3158]
                "rOF": 0.6,             # Decomposition rate constant [cite: 3158]
                "hOF": 0.4              # Humification rate of microbial biomass [cite: 3158]
            },
            "compost": {
                "CNROF": 11.2,
                "CNBOF": 7.8,
                "pcORG": 0.3406,
                "rOF": 0.04078571,
                "hOF": 0.6894089
            },
            "bagasse": {
                "CNROF": 39.0,
                "CNBOF": 34.6461538,
                "pcORG": 0.4101,
                "rOF": 0.05733333,
                "hOF": -0.0591518
            },
            "Fertisol": {
                "CNROF": 9.0,
                "CNBOF": 7.8,
                "pcORG": 0.36,
                "rOF": 0.03511111,
                "hOF": 0.75
            },
            "Vegegwa": {
                "CNROF": 17.0,
                "CNBOF": 42.2117647,
                "pcORG": 0.289,
                "rOF": 0.04870588,
                "hOF": 0.5306354
            }
        }
        
        for k, v in o_parameters[self.of_type].items():
            self.__setattr__(k, v)
    
    def __init__(self, of_type: str = 'Abflor'):
        """
        Initialize the fertilizer manager.
        
        Parameters
        ----------
        of_type : str, optional
            Type of organic fertilizer, by default 'Abflor'. Options include 
            'Abflor', 'compost', 'bagasse', 'Fertisol', and 'Vegegwa'.
        """
        self.of_type = of_type
        
        self.organic_parameters()
        
        self.SOMap = 0
        self.Cr0OF = 0; self.CrOF = 0; self.CbOF = 0; self.ChOF = 0
        self.dNhumOF = 0; self.dNrOF = 0; self.dNbOF = 0
        self.dNRESOF = 0
        self.wrOF: float = 1.0 / self.CNROF; self.wbOF: float = 1.0 / self.CNBOF
    
    def apply_fertilizer(self, is_applied: bool, of_amount: float, Y: float, L: float, wh: float = 0.1) -> None:
        """
        Process the application and ongoing decomposition of organic fertilizer.

        Parameters
        ----------
        is_applied : bool
            Flag indicating whether organic fertilizer is applied this week.
        of_amount : float
            Amount of organic fertilizer applied.
        Y : float
            Assimilation yield (from plant parameters).
        L : float
            Microbial death rate (from plant parameters).
        wh : float, optional
            N:C ratio of the humus, by default 0.1.
        """

        if is_applied:
            self.SOMap = 0
            self.Cr0OF = of_amount * self.pcORG
        
        if self.Cr0OF > 0:
            self.SOMap += 1
            
            # 3. Call the shared engine
            results = nitrogen_release(
                Cr0=self.Cr0OF, 
                r=self.rOF, 
                Y=Y, 
                L=L, 
                t=self.SOMap, 
                h=self.hOF, 
                wr=self.wrOF, 
                wb=self.wbOF,
                wh=wh
            )
            
            # 4. CRITICAL: Extract both the Mineral N AND the Humus N
            self.dNhumOF = results["dNhum"]  # Goes to Soil SON
            self.dNRESOF = results["dNres"]  # Goes to Soil SMN
            
            # (Optional) Save the rest of the state variables for debugging/tracking
            self.CrOF = results["cr"]
            self.CbOF = results["cb"]
            self.ChOF = results["ch"]
            self.dCrOF = results["dCr"]
            self.dCbOF = results["dCb"]
            self.dChumOF = results["dChum"]
            self.dNrOF = results["dNr"]
            self.dNbOF = results["dNb"]
            
        else:
            # If no fertilizer has been applied yet, ensure fluxes are 0
            self.dNhumOF = 0
            self.dNRESOF = 0


class BananaFertiOrganizer():
    def restart_ferti_template(self) -> None:
        """Reset the internal fertilizer template."""
        self.fert_template = {
            "is_applied": [],
            "n_week": [],
            "q_org": [],
            "minn_f": [],
            "minp_f": [],
            "mink_f": []
        }  
    def __init__(self, planting_date):

        self.planting_date = planting_date
        self.restart_ferti_template()
    
    def create_fert_schedule(self,
        weeks_after_planting: Sequence[int],
        n_amounts: Sequence[float]): 

        for wap,namount in zip(weeks_after_planting, n_amounts):
            self.add_ferti_event(wap, namount)
        
        ferti_events = self.fert_template
        self.restart_ferti_template()
        return ferti_events
        

    def add_ferti_event(self, week_after_planting, n_amount):
        n_amount = (n_amount + [0] * (3 - len(n_amount)))[:3] if isinstance(n_amount, list) else [n_amount, 0, 0]

        self.fert_template['is_applied'].append(True)
        self.fert_template['n_week'].append(week_after_planting)
        for ftype,amount in zip(['n', 'p' , 'k'], n_amount): self.fert_template[f'min{ftype}_f'].append(amount)


    def schedule_repeated_applications(self, application_interval: int, n_weeks: int, n_amount: float):
        """
        Schedule fertilization applications at regular intervals over a crop cycle.

    Parameters
    ----------
        application_interval : int
            Number of weeks between each fertilization application.
        n_weeks : int
            Total number of weeks to schedule fertilization over.
        n_amount : float or list of float
            Fertilizer amount(s). Scalar for N only, or list of up to 3 
            elements [N, P, K]. Missing elements are filled with 0.

        Returns
        -------
        dict
            Fertilizer event template with keys: is_applied, n_week, 
            minn_f, minp_f, mink_f.
        """
        weeks = list(range(0, n_weeks, application_interval))
        return self.create_fert_schedule(weeks, [n_amount] * len(weeks))
