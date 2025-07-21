import numpy as np

class Filter:

    def SKLP(self, params, freq, pure=False):
        """Sallen-Key Low-Pass Filter."""
        w_c, zeta = params[0], params[1]
        
        numerator = w_c ** 2
        denominator = w_c ** 2 + 2 * zeta * w_c * 1j * freq + (1j * freq) ** 2

        if pure:
            return numerator / denominator

        return 20 * np.log10(np.abs(numerator / denominator))

    def SKHP(self, params, freq, pure=False):
        """Sallen-Key High-Pass Filter."""
        w_c, zeta = params[0], params[1]

        numerator = (1j * freq) ** 2
        denominator = w_c ** 2 + 2 * zeta * w_c * 1j * freq + (1j * freq) ** 2

        if pure:
            return numerator / denominator

        return 20 * np.log10(np.abs(numerator / denominator))
    
    def BSKBS(self, params, freq):
        "Buffered Sallen_key BandStop Filter."
        w_c1, zeta_1, w_c2, zeta_2 = params[0], params[1], params[2], params[3]

        total = self.SKLP([w_c1, zeta_1], freq, pure = True) + self.SKHP([w_c2, zeta_2], freq, pure = True)

        return 20 * np.log10(np.abs(total))
    
    filters = {0 : SKLP, 1 : SKHP, 2 : BSKBS}
    num_filters = len(filters)
    names = {0 : "Sallen-Key Low-Pass Filter",
             1 : "Sallen-Key High-Pass Filter",
             2 : "Buffered Sallen-Key Band-Stop Filter"}

    def __init__(self, type, params):
        """Initialize the filter with the type and parameters."""
        self.type = type
        self.params = params
    
    def getName(self):
        """Return the name of the filter type."""
        return self.names[self.type]
    
    def getResponse(self, freq):
        """Calculate the filter response at a given frequency."""
        return self.filters[self.type](self.params, freq)
    
    def getComponents(self):
        """Return the parameters of the filter."""
        return self.params
    
    def getRandomComponents(type, w_c_vals, zeta_vals):
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
    
    def getRandomResponse(type, w_c_vals, zeta_vals, freq):
        """Generate a random filter response."""
        params = Filter.getRandomComponents(type, w_c_vals, zeta_vals)
        return Filter(type, params).getResponse(freq)
