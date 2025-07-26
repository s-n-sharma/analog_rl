import random
import numpy as np
from Filter import Filter

class Generator:
    def __init__(self, MAX_FILTERS = 5, omega_c_range = None, zeta_range = None, stageDecider = None):
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
        
        if stageDecider:
            self.stageDecider = stageDecider
        else:
            self.stageDecider = lambda total_gen: random.randint(1, self.MAX_FILTERS)
    
    def getCircuit(self, freq):
        """Generate a random circuit's response with stageDecider deciding the number of stages."""
        return self.getSingleCircuit(self.stageDecider(self.total_gen), freq)
    
    def getRandomComponents(self, type, w_c_vals, zeta_vals):
        """Generate random parameters for the filter type."""
        if type == 0:
            return [np.random.choice(w_c_vals), np.random.choice(zeta_vals)]
        elif type == 1:
            return [np.random.choice(w_c_vals), np.random.choice(zeta_vals)]
        elif type == 2:
            return [np.random.choice(w_c_vals), np.random.choice(zeta_vals),
                    np.random.choice(w_c_vals), np.random.choice(zeta_vals)]
        else:
            raise ValueError("Invalid filter type specified.")
    
    def getRandomResponse(self, type, w_c_vals, zeta_vals, freq):
        """Generate a random filter response."""
        params = Filter.getRandomComponents(type, w_c_vals, zeta_vals)
        return Filter(type, params).getResponse(freq)
    
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
            ret = ret + self.getRandomResponse(component, self.omega_c_range, self.zeta_range, freq)
        
        self.total_gen += 1
        
        return ret
            


        