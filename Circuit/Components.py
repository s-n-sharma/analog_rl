import numpy as np
from abc import ABC, abstractmethod

# =============================================================================
# Abstract Component Class
# =============================================================================
class Component(ABC):
    """Abstract base class for all circuit components."""
    def __init__(self, name, n1, n2):
        self.name = name
        self.n1 = n1
        self.n2 = n2

    @abstractmethod
    def stamp(self, A, z, node_map, freq):
        """Abstract method to stamp the component's effect on MNA matrices."""
        pass

# =============================================================================
# Concrete Component Classes
# =============================================================================
class Resistor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value = float(value)
        self.g = 1 / self.value

    def stamp(self, A, z, node_map, freq):
        # Get matrix indices for nodes, ignoring ground
        p = node_map.get(self.n1)
        n = node_map.get(self.n2)
        if p is not None:
            A[p, p] += self.g
        if n is not None:
            A[n, n] += self.g
        if p is not None and n is not None:
            A[p, n] -= self.g
            A[n, p] -= self.g

class Capacitor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value = float(value)

    def stamp(self, A, z, node_map, freq):
        # Admittance is frequency-dependent
        w = 2 * np.pi * freq
        yc = 1j * w * self.value
        p = node_map.get(self.n1)
        n = node_map.get(self.n2)
        if p is not None:
            A[p, p] += yc
        if n is not None:
            A[n, n] += yc
        if p is not None and n is not None:
            A[p, n] -= yc
            A[n, p] -= yc

class VoltageSource(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value = float(value)

    def stamp(self, A, z, node_map, freq):
        # Associates a new matrix row/col with this source
        p = node_map.get(self.n1)
        n = node_map.get(self.n2)
        src_index = node_map[self.name] # Special mapping for source current
        if p is not None:
            A[p, src_index] = 1
            A[src_index, p] = 1
        if n is not None:
            A[n, src_index] = -1
            A[src_index, n] = -1
        z[src_index] = self.value

class OpAmp(Component):
    """Ideal Op-Amp, modeled as a VCVS with huge gain."""
    def __init__(self, name, n_plus, n_minus, n_out, n_gnd='0', gain=1e6):
        super().__init__(name, n_out, n_gnd) # Use n1/n2 for output
        self.n_plus = n_plus
        self.n_minus = n_minus
        self.gain = gain

    def stamp(self, A, z, node_map, freq):
        # VCVS stamp for the ideal op-amp model
        p_out = node_map.get(self.n1)
        n_out = node_map.get(self.n2) # Ground
        p_in_plus = node_map.get(self.n_plus)
        n_in_minus = node_map.get(self.n_minus)
        src_index = node_map[self.name]

        # KCL at output node
        if p_out is not None:
            A[p_out, src_index] = 1
        if n_out is not None:
            A[n_out, src_index] = -1

        # VCVS relationship: V(out) = gain * (V(n+) - V(n-))
        A[src_index, p_out] = 1
        if p_in_plus is not None:
            A[src_index, p_in_plus] -= self.gain
        if n_in_minus is not None:
            A[src_index, n_in_minus] += self.gain
