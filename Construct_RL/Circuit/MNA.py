import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from Circuit import Circuit
from Components import Resistor, Capacitor, OpAmp, VoltageSource


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- Option 1: Create a circuit from a netlist file ---
    
    # First, create an example netlist file
    netlist_content = """
* Sallen-Key Low-Pass Filter
Vin vin 0 1
R1 vin n1 10e3
R2 n1 vout 10e3
C1 n1 0 10e-9
C2 vout n1 10e-9
E1 vout n1 vout 0
"""
    with open("sallen_key.txt", "w") as f:
        f.write(netlist_content)

    print("--- 1. Testing circuit created from netlist file ---")
    sallen_key_from_file = Circuit()
    sallen_key_from_file.parse_netlist("sallen_key.txt")
    
    # --- Option 2: Create the same circuit programmatically ---
    print("\n--- 2. Testing circuit created programmatically ---")
    sallen_key_prog = Circuit("Sallen-Key Programmatic")
    sallen_key_prog.add_component(VoltageSource('Vin', 'vin', '0', 1.0))
    sallen_key_prog.add_component(Resistor('R1', 'vin', 'n1', 10e3))
    sallen_key_prog.add_component(Resistor('R2', 'n1', 'vout', 10e3))
    sallen_key_prog.add_component(Capacitor('C1', 'n1', '0', 10e-9))
    sallen_key_prog.add_component(Capacitor('C2', 'vout', 'n1', 10e-9))
    # Note: A Sallen-Key uses a voltage follower op-amp config
    sallen_key_prog.add_component(OpAmp('E1', 'n1', 'vout', 'vout'))

    # --- Analyze and Plot (using the programmatically created circuit) ---
    print("\n--- 3. Analyzing and generating Bode plot ---")
    frequencies = np.logspace(1, 6, 500) # 10 Hz to 1 MHz
    
    # We will get the transfer function V(vout) / V(Vin)
    tf = sallen_key_prog.get_transfer_function('Vin', 'vout', frequencies)

    # Calculate magnitude and phase for plotting
    magnitude_db = 20 * np.log10(np.abs(tf))
    phase_deg = np.angle(tf, deg=True)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_mag.semilogx(frequencies, magnitude_db)
    ax_mag.set_title('Bode Plot from Custom MNA Framework')
    ax_mag.set_ylabel('Magnitude (dB)')
    ax_mag.grid(True, which='both')
    
    ax_phase.semilogx(frequencies, phase_deg)
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (degrees)')
    ax_phase.grid(True, which='both')

    # Display the interactive plot window
    print("✅ Analysis complete. Displaying plot...")
    plt.show()