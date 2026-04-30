import numpy as np

from .utils import generate_lognorm_pool
from .soil import BANANASoilMat
from .management import BANANAFerti, nitrogen_release
from typing import Dict, List, Any
import math

class PlantParameters:
    """
    Contains light interception, growth, decomposition, and physiological parameters for the banana plant.

    Attributes
    ----------
    Ea : float
        Maximun light interception efficiency of banana canopy.
    Ec : float
        Proportion of PAR intercepted.
    kBAN : float
        Extinction coefficient of banana canopy (Turner, 1990).
    Pintmax : float
        Maximum proportion of light intercepted by the canopy at flowering (Turner, 1990).
    sdd_iff : float
        Thermal time interval between floral induction and flowering (°C/day).
    sdd_fh : float
        Thermal time interval between flowering and harvest.
    sdd_pif : float
        Thermal time interval between planting and flowering induction.
    andd : float
        Parameters of finger number as a function of dry biomass at floral induction.
    bndd : float
        Parameters of finger number as a function of dry biomass at floral induction.
    Bunchflo : float
        Bunch dry weight at flowering (g).
    RGR : float
        Relative fruit growth rate (g -1 °C d-1).
    DMfruitmax : float
        Maximal finger dry biomass (g).
    slban : float
        Specific leaf area at flowering (m2/g).
    laiban1 : float
        Initial leaf area index of banana (m2 leaf area per m2 ground area).
    laiban_max : int
        Leaf area index of banana for maximal photosynthetically active radiation intercepted.
    phampe : float
        Proportion of stem within the bunch.
    pc_ban : float
        Percentage of carbon in the banana tree.
    r_ban : float
        Decomposition rate constant of residue banana.
    residue_c_yield : float
        Assimilation yield of residue-C by microbial biomass (Y).
    bm_decomr : float
        Decomposition rate constant of microbial biomass (L).
    cn_r_ban : float
        C/N Ratio of banana residues.
    cn_r_hum : float
        C/N humus.
    cn_r_mbban : float
        C:N ratio of zymogenous microbial biomass (CNBBAN).
    h_ban : float
        Humification rate of microbial biomass.
    wr_ban : float
        N:C ratio for equations simplification.
    wb_ban : float
        N:C ratio for microbial biomass equations simplification.
    ZrBAN1 : float
        Partitioning of the banana root exploration of the upper layer.
    ZrBAN2 : float
        Partitioning of the banana root exploration of the lower layer.
    Ksom1 : float
        Mineralization rate of soil organic nitrogen.
    """
    Ea: float = 0.95  # Maximun light interception efficiency of banana canopy
    Ec: float = 0.45 # Proportion of PAR intercepted
    kBAN: float = 0.7 # Extinction coefficient of banana canopy # Turner (1990)
    Pintmax: float = 0.5 # Maximum proportion of light intercepted by the canopy at flowering (Turner, 1990)
    sdd_iff: float = 880 # Thermal time interval between floral induction and flowering -> Measured with data from Rapetti. (2022) °C/day 
    sdd_fh: float = 750.0 # Thermal time interval between flowering and harvest -> Dorel et al. (2016)
    sdd_pif: float = 1451.0 # Thermal time interval between planting and flowering induction -> Measured with BS data
    
    andd: float = 0.0136 ## Parameters of finger number as a function of dry biomass at floral induction
    bndd: float = 151.51 ##Parameters of finger number as a function of dry biomass at floral induction
    
    Bunchflo: float = 644 # Bunch dry weight at flowering
    RGR: float = 0.321 # relative fruit growth rate # g -1 °C d-1
    
    DMfruitmax: float = 35 # Maximal finger dry biomass # g

    slban: float = 0.018 # Specific leaf area at flowering m2/g
    laiban1: float = 0.1 # Initial leaf area index of banana m2 leaf area per m2 ground area
    laiban_max: int = 7 # Leaf area index of banana for maximal photosynthetically active radiation intercepted m2 leaf area per m2 ground area. Measured with data from Ruillé et al. (2023)
    
    phampe: float = 0.06 #Proportion of stem within the bunch
    pc_ban: float = 0.42 #Percentage of carbon in the banana tree
    
    r_ban: float = 0.38 # Decomposition rate constant of residue banana
    
    residue_c_yield: float = 0.62 #Assimilation yield of residue-C by microbial biomas Y
    bm_decomr: float = 0.0076# Decomposition rate constant of microbial biomass L
    
    cn_r_ban: float = 18.3 #C/N Ratio of banana residues (Experiment B)
    cn_r_hum: float = 1/10 # : C/N humus (wh - > r code)
    cn_r_mbban: float = 7.8 if cn_r_ban < 14 else 30.1 - (275/cn_r_ban) # C:N ratio of zymogenous microbial biomass  (CNBBAN)
    
    h_ban: float = 1 - ((0.91 * cn_r_ban) / 16.2 + cn_r_ban) #Humification rate of microbial biomass
    
    wr_ban: float = 1/cn_r_ban #N:C ratio for equations simplification
    wb_ban: float = 1/cn_r_mbban
    
    ZrBAN1: float = 0.5 # Partitioning of the banana root exploration of the upper layer
    ZrBAN2: float = 0.5 # Partitioning of the banana root exploration of the lower layer

    Ksom1: float = 0.0002 # Mineralization rate of soil organic nitrogen
    psk: float = 0.3      # Proportion of biomass allocated to sucker

class BananaCycle(PlantParameters):
    """
    Represents a single plant generation (mother, sucker).

    Attributes
    ----------
    cycle : int
        The generation index of the plant.
    sdd : float
        Thermal time accumulated.
    sdd_pss : float
        Thermal time interval between planting/emergence and sucker emergence.
    laiban : float
        Leaf area index.
    ban_biomass : float
        Total above-ground dry biomass.
    veg_biomass : float
        Aboveground vegetative dry biomass.
    bun_biomass : float
        Dry biomass of banana bunch.
    rac_biomass : float
        Root biomass.
    stress : float
        Nitrogen stress coefficient.
    sominiflo : int
        Indicator of floral induction (1 if true, 0 otherwise).
    recolte : int
        Indicator of harvest (1 if true, 0 otherwise).
    reject_triggered : bool
        Prevents spawning multiple suckers.
    reject : int
        Indicator whether the plant is rejected.
    CrBAN : float
        Carbon in the residue pool.
    dNRESBAN : float
        Mineral N from banana residue mineralization.
    """
    def __init__(self, cycle_id: int, sdd_pss: float):
        self.cycle = cycle_id
        self.sdd = 0
        self.sdd_pss = sdd_pss # Thermal time interval between planting/ emergence and sucker emergence in mat ‘m’ for banana plant in cycle ‘c’
        self.laiban = 0.1 if self.cycle == 1 else 0
        self.ban_biomass = 10.0 if self.cycle == 1 else 0.0 # Total above-ground dry biomass of banana
        self.veg_biomass = 10.0 if self.cycle == 1 else 0.0 # Aboveground vegetative dry biomass in mat
        self.bun_biomass = 0.0 # Dry biomass of banana bunch in mat 'm'
        self.rac_biomass = 0.0 # Root biomass
        
        self.stress = 1.0 # N stress coefficient
        self.sominiflo = 0 # Indicator of floral induction (1 if floral induction has occurred, 0 otherwise)
        self.sdd_post_iniflo = 0 # Thermal time since flowering
        self.recolte = 0 # Indicator of harvest (1 if harvest has occurred, 0 otherwise)
        self.som_recolte = 0
        
        self.somfloraison = 0       # Indicator of flowering (1 if true, 0 otherwise)
        self.sdd_post_floraison = 0 # Thermal time accumulated post flowering

        self.reject_triggered = False # prevents spawning multiple suckers
        self.reject = 0 # Indicator of whether the plant is rejected (1) or not (0)
        self.ndd = 0 # Number of fingers
        self.dmfruit = 0 # Dry biomass of fruit in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        
        self.dDMBANtot = 0 # Total newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        self.dDMBAN = 0 # Net newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        self.alloc_bun = 0 # Dry biomass allocated to the banana bunch in mat ‘m’ for banana plant
        self.received_biomass = 0 # Dry biomass received from the parent plant # allocfromPM
        self.alloc_suc = 0.0 
        # Residues
        self.Cr0BAN = 0 # Initial carbon in the residue pool at harvest
        self.CrBAN = 0 # Carbon in the residue pool
        self.CbBAN = 0 # Carbon in the microbial biomass pool
        self.ChBAN = 0 # Carbon in the humus pool
        self.dNhumBAN = 0 # N change in humified soil organic matter due to banana residues
        self.dNrBAN = 0 # N change in banana residues
        self.dNbBAN = 0 # N change in soil microbial biomass due to banana residues
        self.dNRESBAN = 0 # Mineral N from banana residue mineralization
        
    def update_phenology(self, temperature: float) -> None: 
        """
        Update the phenological state of the plant based on the accumulated thermal time and stress factors.
        
        Parameters
        ----------
        temperature : float
            Daily temperature for thermal time accumulation.

        Notes
        -----
        Equation 1. Banana N. Ruillé et al. (2025)
        """
        stress_factor_value = self.stress if self.sominiflo < 1 else 1.0
        stress_factor_value = max(0.1, stress_factor_value) # Ensure that the stress factor does not drop below 0.1
        if self.sdd < 0: stress_factor_value = 1.0 # No stress before planting
        self.sdd += max(0, temperature) * stress_factor_value # Accumulate thermal time with stress factor
        self.sominiflo = 1 if self.sdd >= self.sdd_pss else 0 # Floral induction occurs when accumulated thermal time exceeds the threshold
        self.sdd_post_iniflo = self.sdd_post_iniflo + temperature if self.sominiflo == 1 else 0 # Accumulate thermal time since floral induction    
        self.somfloraison = self.somfloraison + 1 if self.sdd_post_iniflo >= self.sdd_iff else 0 # floral induction and flowering
        self.sdd_post_floraison = self.sdd_post_floraison + temperature if self.somfloraison == 1 else 0 # Accumulate thermal time post flowering
        
        self.recolte = 1 if self.sdd_post_floraison >= self.sdd_fh else 0 # Harvest occurs when accumulated thermal time post flowering exceeds the threshold
        self.som_recolte = self.som_recolte + 1 if self.recolte == 1 else 0
    
    def update_biomass_and_allocation(self, temperature: float, surface_area: float) -> None:
        """
        Update the biomass accumulation and allocation to different plant parts.

        Parameters
        ----------
        temperature : float
            Daily temperature.
        surface_area : float
            Surface area available for the plant (m2).
        """
        if self.recolte == 1:
            self.ban_biomass = 0.0
            self.veg_biomass = 0.0
            self.bun_biomass = 0.0
            self.laiban = 0.0
        
            
        self.ndd = self.andd if self.sominiflo == 1 else self.andd * self.ban_biomass + self.bndd  # aNDD×Biomass at floral induction+bNDD A1.1. Ruillé et al. (2025)
        self.dmfruit = self.bun_biomass/self.ndd if (self.somfloraison >= 1 and self.recolte < 1) and self.ndd > 0 else 0.0
        
        if self.sominiflo < 1 or self.recolte >= 1: 
            alloc_bun = 0.0
        elif self.sominiflo >= 1 and self.somfloraison < 1:
            alloc_bun = self.Bunchflo / self.sdd_iff * temperature
        else:
            alloc_bun = self.RGR  * temperature * (1 - (self.dmfruit / self.DMfruitmax)) * self.ndd
        
        alloc_bun = min(alloc_bun, self.dDMBAN) # The biomass allocated to the bunch cannot exceed the total newly formed biomass
        
        if self.sominiflo < 1:
            alloc_veg = self.dDMBAN + self.received_biomass # Before floral induction, all biomass goes to vegetative growth
        else:
            alloc_veg = self.dDMBAN - alloc_bun + self.received_biomass    
        
        self.received_biomass = 0.0
        
        ## accumulate biomass
        self.ban_biomass += self.dDMBAN  ## equation 9 Ruillé et al. (2025)
        
        self.veg_biomass += alloc_veg + self.alloc_suc # equation 10 Ruillé et al. (2025), there is modification included alloc_suc which is the biomass received from the parent plant (sucker)
        self.bun_biomass += alloc_bun # equation 11 Ruillé et al. (2025)
        senBan = 0.25 if self.somfloraison >= 1 else 0.013
        plv = 0.51 if self.sominiflo < 1 else 0.3
        
        ## BANANA Leaf area index                
        if self.somfloraison < 1:
            prod1 = alloc_veg * plv * self.slban * ((self.laiban_max - self.laiban) / self.laiban_max) / surface_area # equation 5 Ruillé et al. (2025)
            self.laiban = self.laiban + prod1  - self.laiban * senBan 
        else:
            self.laiban = self.laiban - (self.laiban * senBan)
            
    
    def calculate_mineralN_fromBANresidues(self) -> None:
        """
        Calculate mineral Nitrogen generation from banana plant residues after harvest.
        """
        if self.som_recolte == 1:
            self.Cr0BAN = (self.veg_biomass + self.bun_biomass * self.phampe) * self.pc_ban 
        
        # 2. Process decomposition for week 1 and onwards (>= 1)
        if self.som_recolte >= 1:
            
            # Call the shared engine
            results = nitrogen_release(
                Cr0=self.Cr0BAN, 
                r=self.r_ban, 
                Y=self.residue_c_yield, 
                L=self.bm_decomr, 
                t=self.som_recolte, 
                h=self.h_ban, 
                wr=self.wr_ban, 
                wb=self.wb_ban
            )
            
            # Map the results back to the object's state
            self.CrBAN = results["cr"]
            self.CbBAN = results["cb"]
            self.ChBAN = results["ch"]
            
            self.dCrBAN = results["dCr"]
            self.dCbBAN = results["dCb"]
            self.dchumban = results["dChum"]
            
            self.dNrBAN = results["dNr"]
            self.dNbBAN = results["dNb"]
            self.dnhumban = results["dNhum"]
            
            self.dNRESBAN = results["dNres"]   
            


class BananaMat_cycles(PlantParameters):
    """
    Represents a banana mat managing multiple plant cycles and soil state.
    """
    def __init__(self, mat_id: int, density: float, pool_sdd: np.ndarray, init_soil_parameters: Dict[str, float]):
        """
        Initialize the banana mat cycles manager.

        Parameters
        ----------
        mat_id : int
            Identifier for the mat.
        density : float
            Planting density (number of plants per hectare).
        pool_sdd : np.ndarray
            Thermal time interval between planting/emergence and sucker emergence.
            Stochastically defined with a lognormal distribution.
        init_soil_parameters : Dict[str, float]
            Initial soil parameters for the mat.
        """
        self.mat_id = mat_id
        self.surface_area = 10000.0 / density # m2 per mat
        
        self.cycles: List[BananaCycle] = [BananaCycle(cycle_id=1, sdd_pss=1755.0)]
        self.soil = BANANASoilMat(mat_id, **init_soil_parameters)
        self.ferti = BANANAFerti()
        
        self.pool_sdd = pool_sdd
        
        
    def update_mat(self, week: int, temperature: float, radiation: float, rain: float, et: float, is_fertilizer_applied: bool, of_amount: float, minf_amount: float) -> None:
        """
        Update the state of the mat for a given week based on environmental conditions and management practices.

        Parameters
        ----------
        week : int
            The current week of the simulation.
        temperature : float
            Average or accumulated temperature for the week.
        radiation : float
            Solar radiation for the week.
        rain : float
            Rainfall amount for the week.
        et : float
            Evapotranspiration for the week.
        is_fertilizer_applied : bool
            True if organic fertilizer is applied in this week.
        of_amount : float
            Amount of organic fertilizer applied.
        minf_amount : float
            Amount of mineral fertilizer applied.
        """
        mat_lai = sum(c.laiban for c in self.cycles if c.recolte == 0) # Calculate the total leaf area index of the mat by summing the leaf area index of all cycles that have not been harvested
        
        pari_ban = self.Ea * self.Ec * radiation * (1 - np.exp(-self.kBAN * mat_lai)) # PAR intercepted by the canopy
        
        sum_dNResBAN = 0 # total nitrogen mineralized from banana residues across all cycles in the mat
        sum_dNHumBAN = 0 # total nitrogen humified from banana residues across all cycles in the mat
        sum_dDMBANtot = 0 # total newly formed dry biomass across all cycles in the mat
        
        
        for i, cycle in enumerate(self.cycles):        
            cycle.update_phenology(temperature)
            
            eb_ban = 1.9
            if cycle.sominiflo < 1 and cycle.cycle == 1:
                eb_ban = 1.2
            elif cycle.sominiflo >= 1 and cycle.cycle == 1:
                eb_ban = 1.2
            elif cycle.sominiflo < 1:
                eb_ban = 1.9
            else:
                eb_ban = 2.5
            
            lai_banprod = cycle.laiban / mat_lai if mat_lai > 0 else 0 # Proportion of the mat's leaf area index that belongs to the current cycle
            ## biomass production
            cycle.dDMBANtot = (eb_ban * pari_ban * self.surface_area * (1 - cycle.recolte)) * cycle.stress * lai_banprod # Total newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’ (g DM/week) ## equation 3 Ruillé et al. (2025)            
            sum_dDMBANtot += cycle.dDMBANtot
            
            # allocation to sucker
            cycle.alloc_suc = cycle.psk * cycle.dDMBANtot if cycle.reject >= 1 else 0
            cycle.dDMBAN = cycle.dDMBANtot - cycle.alloc_suc
            #allocfromPM
            if cycle.reject >= 1 and i + 1 < len(self.cycles):
                self.cycles[i+1].received_biomass = cycle.alloc_suc
            
            cycle.update_biomass_and_allocation(temperature, self.surface_area)
            cycle.calculate_mineralN_fromBANresidues()
            
            sum_dNResBAN += cycle.dNRESBAN
            sum_dNHumBAN += cycle.dNhumBAN
        
        new_cycles = []
        for cycle in self.cycles:
            if cycle.reject == 1 and not cycle.reject_triggered:
                cycle.reject_triggered = True
                new_sdd = np.random.choice(self.pool_sdd)
                new_cycles.append(BananaCycle(cycle.cycle + 1, new_sdd))
        
        self.cycles.extend(new_cycles)
        total_biomass = sum(c.ban_biomass for c in self.cycles)
        pnBAN = (4.78 * math.pow(total_biomass,-0.13))/100 if total_biomass > 0 else 0
        dNBAN = sum_dDMBANtot * pnBAN
        
        dNBAN_1 = dNBAN * self.ZrBAN1
        dNBAN_2 = dNBAN * self.ZrBAN2
    
        self.ferti.apply_fertilizer(is_fertilizer_applied, of_amount, self.residue_c_yield, self.bm_decomr)        
        
        # Water balance
        kc = -0.0487 * mat_lai**2 + 0.3925 * mat_lai + 0.4235
        
        etr = et * kc
        et1 = etr * self.ZrBAN1
        et2 = etr * self.ZrBAN2
        
        wal1 = max(0, rain - et1 - (self.soil.SW1 - self.soil.wsol1))
        wal = max(0, wal1 - et2 - (self.soil.SW2 - self.soil.wsol2))
        
        self.soil.wsol1 = max(0, self.soil.wsol1 + rain - et1 - wal1)
        self.soil.wsol2 = max(0, self.soil.wsol1 + wal1 + et2 - wal)
        
        # leaching
        nl1 = self.soil.SMN1 * (1 -np.exp(-self.soil.kl1 * wal1 / self.soil.SW1))
        nal = self.soil.SMN2 + nl1
        nl = max(0, nal * (1-np.exp(-self.soil.kl2 * wal / self.soil.SW2)))
        
        # soil organic matter mineralization 
        
        mos = self.soil.SON * (-np.exp(self.Ksom1 * (week + 1)) + np.exp(-self.Ksom1 * week))
        
        self.soil.SON = self.soil.SON - mos + sum_dNHumBAN + self.ferti.dNhumOF
        
        # uptake
        
        uban1 = min(dNBAN_1, self.soil.SMN1)
        uban2 = min(dNBAN_2, self.soil.SMN2)
        uban = uban1 + uban2
        
        #
        self.soil.SMN1 = max(0.0, self.soil.SMN1 + mos - nl1 + minf_amount - uban1 + sum_dNResBAN + self.ferti.dNRESOF)
        self.soil.SMN2 = max(0.0, self.soil.SMN2 - nl + nl1 - uban2)        
        self.soil.SMN = self.soil.SMN1 + self.soil.SMN2

        
        mat_stress = uban / dNBAN if sum_dDMBANtot > 0 and dNBAN > 0 else 1.0

        for cycle in self.cycles:
            cycle.stress = mat_stress
            
     
     
class BANANAField:
    """
    Simulates a full banana field consisting of multiple mats.

    Parameters
    ----------
    nban : int, optional
        Number of banana mats to simulate in the field, by default 40.
    density : int, optional
        Planting density (mats per hectare), by default 1300.0.

    Attributes
    ----------
    nban : int
        Number of mats in the field.
    pool_sdd : np.ndarray
        Pool of possible sum-of-degree-days generated for sucker emergence.
    mats : List[BananaMat]
        List of BananaMat objects representing the field.
    """
    def __init__(self, nban: int = 40, density: float = 1300.0, init_soil_parameters: Dict[str, float] = None):
        if init_soil_parameters is None:
            init_soil_parameters = {}
        self.nban = nban
        self.density = density   

        self.pool_sdd: np.ndarray = generate_lognorm_pool(7.102693, 0.1240221)
        self.flowering_delay_weeks = generate_lognorm_pool(1.0, 0.43) # Allows you to create the week offset (flowering), just for the first cycle 
        self.mats: List[BananaMat_cycles] = [BananaMat_cycles(i, density, self.pool_sdd, init_soil_parameters) for i in range(nban)]
    
    
    def simulate(
        self,
        nb_weeks: int,
        weather_data: List[Dict[str, float]],
        ferti_schedule: List[Dict[str, Any]]
        ) -> List[Dict[str, float]]:
        """
        Runs the simulation for a specified number of weeks.

        Parameters
        ----------
        nb_weeks : int
            Number of weeks to simulate.
        weather_data : List[Dict[str, float]]
            List of dictionaries containing weather data (temp, rad, rain, et) per week.
        ferti_schedule : List[Dict[str, Any]]
            List of dictionaries defining fertilizer applications per week.

        Returns
        -------
        List[Dict[str, float]]
            A history (list of daily logs) over the simulated weeks containing averages of SMN and Biomass.
        """
        history = []

        for week in range(nb_weeks):
            w = weather_data[week]
            f = ferti_schedule[week]

            total_smn:float = 0.0
            total_biomass_g:float = 0.0
            total_fruit_g: float = 0.0

            for mat in self.mats:
                delay_f = int(np.random.choice(self.flowering_delay_weeks))
                if week<int(delay_f): mat.stress = 0
                mat.update_mat(
                        week = week, temperature = w['dtt'],
                        radiation = w['srad'], rain = w['rain'],
                        et = w['etr'], is_fertilizer_applied = f['application'], of_amount = f['q_org'], 
                        minf_amount = f['min_f'])
                
                total_smn += mat.soil.SMN
                total_biomass_g += sum(c.ban_biomass for c in mat.cycles)
                total_fruit_g += sum(c.bun_biomass for c in mat.cycles)

            avg_biomass_g_per_mat = total_biomass_g / self.nban
            avg_fruit_g_per_mat  = total_fruit_g / self.nban

            history.append({
                'Week': week,
                'Avg_SMN_kg_ha': total_smn / self.nban,
                'Avg_Bioamass_g_mat': avg_biomass_g_per_mat,
                'Avg_Fruit_g_mat': avg_fruit_g_per_mat
            })

        return history 
