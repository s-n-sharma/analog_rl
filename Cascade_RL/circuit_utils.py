# circuit_utils.py
import numpy as np
import random

# --- Filter Functions (formerly methods of Filter class) ---

# A small epsilon to prevent log(0) issues
EPSILON = 1e-10

def lp_filter(params, freq, pure=False):
    """Low-Pass filter"""
    w_0 = params[0]
    
    numerator = 1.0
    # Add a small value to denominator to prevent division by zero or extremely large values
    denominator = 1j * freq / w_0 + 1.0 

    if pure:
        return numerator / denominator
    return 20 * np.log10(np.abs(numerator / denominator) + EPSILON)

def hp_filter(params, freq, pure=False):
    """High-pass filter"""
    w_0 = params[0]

    numerator = 1j * freq / w_0
    # Add a small value to denominator to prevent division by zero or extremely large values
    denominator = 1j * freq / w_0 + 1.0

    if pure:
        return numerator / denominator
    return 20 * np.log10(np.abs(numerator / denominator) + EPSILON)

def bskbs_filter(params, freq, pure=False):
    """Buffered Sallen-Key Band-Stop Filter (using params[0] as w_c1 and params[1] as w_c2)"""
    w_c1, zeta_1 = params[0], params[1]
    w_c2, zeta_2 = params[2], params[3] # This expects 4 params

    numerator_lp = w_c1 ** 2
    denominator_lp = w_c1 ** 2 + 2 * zeta_1 * w_c1 * 1j * freq + (1j * freq) ** 2
    
    numerator_hp = (1j * freq) ** 2
    denominator_hp = w_c2 ** 2 + 2 * zeta_2 * w_c2 * 1j * freq + (1j * freq) ** 2
    
    # Using the standard Sallen-Key forms
    # Note: If you want to use the simpler LP/HP functions, you might need to reconsider
    # how BSKBS is defined. Assuming the original SKLP/SKHP logic from your Filter.py
    # for the `bskbs_filter` definition.
    
    # Adding EPSILON to denominators to prevent division by zero or very small numbers
    response_lp_pure = numerator_lp / (denominator_lp + EPSILON)
    response_hp_pure = numerator_hp / (denominator_hp + EPSILON)

    total = response_lp_pure + response_hp_pure
    return 20 * np.log10(np.abs(total) + EPSILON) # Add EPSILON here too

# For the `SUBCIRCUITS` you provided, it seems you intend to use the simpler RC filters (LP, HP, BRCBS)
# Let's adjust the names to reflect that for clarity.
def brcbs_filter(params, freq, pure=False):
    """Buffered RC Band Stop Filter"""
    w_0_a, w_0_b = params[0], params[1]
    
    # Call the simpler LP and HP functions (pure output)
    total = lp_filter([w_0_a], freq, pure=True) + hp_filter([w_0_b], freq, pure=True)
    return 20 * np.log10(np.abs(total) + EPSILON)

# Define the available filter functions and their parameter counts
FILTER_FUNCTIONS = {
    "LP": lp_filter,
    "HP": hp_filter,
    "BRCBS": brcbs_filter, # Using the simpler RC Band Stop
    # "SKLP": sklp_filter, # If you want to use Sallen-Key filters, define them separately
    # "SKHP": skhp_filter,
    # "BSKBS": bskbs_filter, # This one requires 4 params (2 w_c, 2 zeta)
}

# --- Utility Functions for Random Circuit Generation ---

def get_random_filter_params(filter_name, w_c_range, zeta_range=None):
    """Generate random parameters for a given filter type."""
    if filter_name == "LP":
        return [np.random.choice(w_c_range)]
    elif filter_name == "HP":
        return [np.random.choice(w_c_range)]
    elif filter_name == "BRCBS":
        return [np.random.choice(w_c_range), np.random.choice(w_c_range)]
    # Add other filter types and their parameter generation logic as needed
    elif filter_name == "SKLP" or filter_name == "SKHP":
        return [np.random.choice(w_c_range), np.random.choice(zeta_range)]
    elif filter_name == "BSKBS": # This one requires 4 params (2 w_c, 2 zeta)
        return [np.random.choice(w_c_range), np.random.choice(zeta_range),
                np.random.choice(w_c_range), np.random.choice(zeta_range)]
    else:
        raise ValueError(f"Invalid filter type specified: {filter_name}")

def generate_random_circuit_response(num_stages, freq, w_c_range, zeta_range, available_filters):
    """Generate a random circuit's response by cascading multiple random filters."""
    ret = np.zeros(len(freq), dtype=np.float64)
    
    # Randomly select filter names from available_filters
    chosen_filter_names = random.choices(list(available_filters.keys()), k=num_stages)

    for filter_name in chosen_filter_names:
        filter_func = available_filters[filter_name]
        params = get_random_filter_params(filter_name, w_c_range, zeta_range)
        ret += filter_func(params, freq)
    
    return ret

# --- Configuration for Filter Parameters Ranges ---
# These can be externalized or passed as arguments if needed
DEFAULT_OMEGA_C_RANGE = np.logspace(1, 6, 125) # More realistic range, e.g., 10 to 1MHz
# Ensure zeta_range is defined if using Sallen-Key filters
DEFAULT_ZETA_RANGE = np.concatenate((np.linspace(0.1, 0.4, 25), 
                                     np.linspace(0.5, 1.5, 75), 
                                     np.linspace(1.6, 2, 25)))