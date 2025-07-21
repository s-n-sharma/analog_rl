import random
import numpy as np
from Filter import Filter

class Generator:
    def __init__(self, MAX_FILTERS = 5, omega_c_range = None, zeta_range = None):
        """Initialize the generator"""
        self.total_gen = 0
        self.MAX_FILTERS = MAX_FILTERS

        if omega_c_range:
            self.omega_c_range = omega_c_range
        else:
            self.omega_c_range = np.logspace(1e2, 1e6, 125)  # Default range from 100 Hz to 1 MHz

        if zeta_range:
            self.zeta_range = zeta_range
        else:
            middle_zeta_range = np.linspace(0.5, 1.5, 75)
            upper_zeta_range = np.linspace(1.6, 2, 25)
            lower_zeta_range = np.linspace(0.1, 0.4, 25)

            self.zeta_range = np.concatenate((lower_zeta_range, middle_zeta_range, upper_zeta_range))
    
    def getCircuit(self, stageDecider, freq):
        """Generate a random circuit's response with stageDecider deciding the number of stages."""
        return self.getSingleCircuit(stageDecider(self.total_gen), freq)
    
    def getSingleCircuit(self, numStages, freq):
        """Generate a single circuit's response with numStages stages for the specified frequencies."""
        if numStages > self.MAX_FILTERS:
            raise ValueError(f"Cannot generate more than {self.MAX_FILTERS} filters in a single circuit.")
        
        if numStages < 1:
            raise ValueError("Number of components must be at least 1.")
        
        # Generate a random assortment of stages

        components = random.choices(list(range(Filter.num_filters)), k=numStages)

        ret = np.zeros(len(freq), dtype=np.float64)

        for component in components:
            ret = ret + Filter.getRandomResponse(component, self.omega_c_range, self.zeta_range, freq)
        
        self.total_gen += 1
        
        return ret
            



        

        
        self.total_gen += 1


        omega_c = np.random.choice(self.omega_c_range, numComponents)
        zeta = np.random.choice(self.zeta_range, numComponents)

        return omega_c, zeta


        

        