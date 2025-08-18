import numpy as np
import random

# A small epsilon to prevent issues with log(0) or division by zero
EPSILON = 1e-10

# --- NEW: Define a safe range for frequency parameters ---
# This prevents the optimizer from testing values that cause numerical explosions.
# The range is extremely wide (0.01 Hz to 10 GHz) and should not limit the solution.
MIN_W0 = 1e-2
MAX_W0 = 1e10

def lp_filter(params, freq, pure=False):
    """First-order RC Low-Pass filter."""
    # --- FIX: Clip the parameter to a safe range ---
    w_0 = np.clip(params[0], MIN_W0, MAX_W0)
    
    s = 1j * freq
    response = 1 / (s / w_0 + 1)

    if pure:
        return response
    return 20 * np.log10(np.abs(response) + EPSILON)

def hp_filter(params, freq, pure=False):
    """First-order RC High-pass filter."""
    # --- FIX: Clip the parameter to a safe range ---
    w_0 = np.clip(params[0], MIN_W0, MAX_W0)

    s = 1j * freq
    x = s / w_0
    # Using the previously implemented stable form, now with clipped w_0
    response = 1 / (1 + 1 / (x + EPSILON))

    if pure:
        return response
    return 20 * np.log10(np.abs(response) + EPSILON)

def brcbs_filter(params, freq, pure=False):
    """Buffered RC Band-Stop Filter made by summing a LP and HP filter."""
    # --- FIX: Clip parameters to a safe range ---
    w_0_lp = np.clip(params[0], MIN_W0, MAX_W0)
    w_0_hp = np.clip(params[1], MIN_W0, MAX_W0)
    
    total_response = lp_filter([w_0_lp], freq, pure=True) + hp_filter([w_0_hp], freq, pure=True)
    
    if pure:
        return total_response
    return 20 * np.log10(np.abs(total_response) + EPSILON)

# Define the available filter functions for random generation
FILTER_FUNCTIONS = {
    "LP": lp_filter,
    "HP": hp_filter,
    "BRCBS": brcbs_filter
}

# --- Utility Functions for Random Circuit Generation ---

def get_random_filter_params(filter_name, w_c_range, zeta_range):
    """Generates random parameters for a given filter type."""
    if filter_name in ["LP", "HP"]:
        return [np.random.choice(w_c_range)]
    elif filter_name == "BRCBS":
        return [np.random.choice(w_c_range), np.random.choice(w_c_range)]
    else:
        raise ValueError(f"Invalid filter type specified for random generation: {filter_name}")

def generate_random_circuit_response(num_stages, freq, w_c_range, zeta_range, available_filters):
    """
    Generates a random circuit's response by cascading multiple random filters.
    Returns the final response and the list of ground truth filters used.
    """
    cascaded_response = np.zeros(len(freq), dtype=np.float64)
    ground_truth_filters = []
    
    chosen_filter_names = random.choices(list(available_filters.keys()), k=num_stages)

    for filter_name in chosen_filter_names:
        filter_func = available_filters[filter_name]
        params = get_random_filter_params(filter_name, w_c_range, zeta_range)
        ground_truth_filters.append({'name': filter_name, 'params': params})
        cascaded_response += filter_func(params, freq)
    
    return cascaded_response, ground_truth_filters

# --- Default Configuration for Parameter Ranges ---
DEFAULT_OMEGA_C_RANGE = np.logspace(2, 5, 200) # Range from 100 Hz to 100 kHz
DEFAULT_ZETA_RANGE = np.linspace(0.1, 2.0, 50)