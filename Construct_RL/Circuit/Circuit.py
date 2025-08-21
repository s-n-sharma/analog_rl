from Components import Component, Resistor, Capacitor, OpAmp, VoltageSource
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# Circuit Class
# =============================================================================
class Circuit:
    def __init__(self, name="Circuit"):
        self.name = name
        self.components = []
        self._is_prepared = False

    def add_component(self, component):
        self.components.append(component)
        self._is_prepared = False

    def _prepare_mna(self):
        """Identifies nodes and sources to set up matrix dimensions."""
        node_names = set(['0']) # Ground is always present
        vs_names = []
        for comp in self.components:
            node_names.add(comp.n1)
            node_names.add(comp.n2)
            if isinstance(comp, OpAmp):
                node_names.add(comp.n_plus)
                node_names.add(comp.n_minus)
            if isinstance(comp, (VoltageSource, OpAmp)):
                vs_names.append(comp.name)
        
        # Create the node-to-matrix-index mapping
        self.node_map = {name: i for i, name in enumerate(sorted(list(node_names - {'0'})))}
        
        # Add voltage source current variables to the map
        offset = len(self.node_map)
        for i, name in enumerate(vs_names):
            self.node_map[name] = offset + i
            
        self.matrix_size = len(self.node_map)
        self._is_prepared = True

    def get_transfer_function(self, input_source_name, output_node_name, freqs):
        if not self._is_prepared:
            self._prepare_mna()

        results = []
        for f in freqs:
            # Create fresh matrices for each frequency
            A = np.zeros((self.matrix_size, self.matrix_size), dtype=np.complex128)
            z = np.zeros(self.matrix_size, dtype=np.complex128)

            # Populate matrices by stamping each component
            for comp in self.components:
                # Set input source value to 1 for transfer function
                if isinstance(comp, VoltageSource) and comp.name == input_source_name:
                    comp.value = 1.0
                comp.stamp(A, z, self.node_map, f)
            
            # Solve the system
            try:
                x = np.linalg.solve(A, z)
                output_index = self.node_map.get(output_node_name)
                if output_index is not None:
                    results.append(20*np.log10(np.abs(x[output_index])))
                else:
                    results.append(1) # Output is ground
            except np.linalg.LinAlgError:
                results.append(1.0)
        
        return np.array(results)


    # =============================================================================
    # Parser Function
    # =============================================================================
    def parse_netlist(self, filepath):
        """Parses a .txt netlist file and returns a Circuit object."""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                
                parts = line.split()
                name = parts[0]
                comp_type = name[0].upper()
                
                if comp_type == 'R':
                    self.add_component(Resistor(name, parts[1], parts[2], parts[3]))
                elif comp_type == 'C':
                    self.add_component(Capacitor(name, parts[1], parts[2], parts[3]))
                elif comp_type == 'V':
                    self.add_component(VoltageSource(name, parts[1], parts[2], parts[3]))
                elif comp_type == 'E': # Op-Amp
                    # E1 n+ n- out gnd
                    self.add_component(OpAmp(name, parts[1], parts[2], parts[3]))
    
    