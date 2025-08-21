import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# Component Base & Subclasses (for a complete, runnable example)
# =============================================================================
class Component:
    """Base class for all circuit components."""
    def __init__(self, name, n1, n2):
        self.name = name
        self.n1 = n1
        self.n2 = n2

    def stamp(self, A, z, node_map, f):
        """Placeholder for MNA stamping logic."""
        pass

class Resistor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value_str = value
        # Simple parsing for float value
        try:
            self.value = float(value.replace('k', 'e3').replace('M', 'e6'))
        except ValueError:
            self.value = 0

class Capacitor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value_str = value
        try:
            self.value = float(value.replace('u', 'e-6').replace('n', 'e-9').replace('p', 'e-12'))
        except ValueError:
            self.value = 0

class VoltageSource(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, n1, n2)
        self.value_str = value
        try:
            self.value = float(value)
        except (ValueError, TypeError):
            self.value = 0

class OpAmp(Component):
    def __init__(self, name, n_plus, n_minus, n_out):
        # n1 and n2 for MNA represent the output voltage source
        super().__init__(name, n_out, '0')
        self.n_plus = n_plus
        self.n_minus = n_minus
        self.n_out = n_out
        self.value_str = "Ideal" # No specific value for visualization

# =============================================================================
# Circuit Class (with new visualize method)
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
                node_names.add(comp.n_out) # n_out is same as n1
            if isinstance(comp, (VoltageSource, OpAmp)):
                vs_names.append(comp.name)
        
        self.node_map = {name: i for i, name in enumerate(sorted(list(node_names - {'0'})))}
        
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
            A = np.zeros((self.matrix_size, self.matrix_size), dtype=np.complex128)
            z = np.zeros(self.matrix_size, dtype=np.complex128)

            for comp in self.components:
                if isinstance(comp, VoltageSource) and comp.name == input_source_name:
                    comp.value = 1.0
                # In a real scenario, you'd implement the stamp methods
                # comp.stamp(A, z, self.node_map, f)
            
            try:
                # Placeholder solve, as stamp methods are not implemented
                # x = np.linalg.solve(A, z) 
                # For now, just append a dummy value
                results.append(complex(0,0))
            except np.linalg.LinAlgError:
                results.append(complex(0,0))
        
        return np.array(results)

    def parse_netlist(self, filepath):
        """Parses a .txt netlist file and adds components to the circuit."""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip().lower() # Make parsing case-insensitive
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
                elif comp_type == 'E': # Op-Amp: E1 n+ n- out
                    self.add_component(OpAmp(name, parts[1], parts[2], parts[3]))

    # =========================================================================
    # NEW VISUALIZATION METHOD
    # =========================================================================
    def visualize(self):
        """Draws the circuit as a graph."""
        if not self.components:
            print("Circuit is empty. Add components before visualizing.")
            return

        G = nx.Graph()
        edge_labels = {}
        opamp_nodes = []

        # Add components as nodes and edges to the graph
        for comp in self.components:
            if isinstance(comp, OpAmp):
                # Represent OpAmp as a central node
                opamp_node_name = comp.name
                opamp_nodes.append(opamp_node_name)
                G.add_node(opamp_node_name)
                # Connect terminals to the OpAmp's central node
                G.add_edge(comp.n_plus, opamp_node_name)
                G.add_edge(comp.n_minus, opamp_node_name)
                G.add_edge(comp.n_out, opamp_node_name)
                # Add labels for the OpAmp terminals
                edge_labels[(comp.n_plus, opamp_node_name)] = '+'
                edge_labels[(comp.n_minus, opamp_node_name)] = '–'
                edge_labels[(comp.n_out, opamp_node_name)] = 'out'
            else:
                # Standard two-terminal components
                G.add_edge(comp.n1, comp.n2)
                edge_labels[(comp.n1, comp.n2)] = f"{comp.name}\n{comp.value_str}"

        # Use a spring layout for positioning nodes
        pos = nx.spring_layout(G, seed=42)

        # Prepare node styling
        regular_nodes = [n for n in G.nodes() if n not in opamp_nodes]
        node_colors = ['#2b2b2b' if n == '0' else '#1f78b4' for n in regular_nodes]

        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Draw regular circuit nodes (circles)
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_color=node_colors, node_size=1000)
        
        # Draw OpAmp nodes (triangles)
        nx.draw_networkx_nodes(G, pos, nodelist=opamp_nodes, node_shape='^', node_color='#ff7f0e', node_size=1500)

        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        # Display the plot
        plt.title(f"'{self.name}' Connectivity Diagram", size=16)
        plt.margins(0.1)
        plt.axis('off')
        plt.show()

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # 1. Create a sample netlist file (e.g., a Sallen-Key low-pass filter)
    netlist_content = """
* Sallen-Key Low-Pass Filter
Vin in 0 1
R1 in 1 1k
R2 1 out 1k
C1 1 2 1u
C2 2 0 1u
E1 2 out out
"""
    with open("sallen_key.txt", "w") as f:
        f.write(netlist_content)

    # 2. Create a Circuit object and parse the file
    my_circuit = Circuit(name="Sallen-Key Filter")
    my_circuit.parse_netlist("sallen_key.txt")

    # 3. Visualize the circuit
    my_circuit.visualize()