import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cma
import random
from tqdm import tqdm

# Import the RLAgent and circuit utilities
# We will define the different Optimizer classes directly in this script
from trainer import RLAgent
import circuit_utils
from trainer import RLAgent, SUBCIRCUITS

# --- Optimizer Versions ---
# We define the different optimizer classes here to ensure we test each model
# with the exact logic it was trained on.

class Optimizer_DerivativeAware:
    """The optimizer that uses a global derivative-aware loss."""
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected:
            p_initial = [1000.0] * num_params_expected
        log_p_initial = np.log10(p_initial)
        def log_objective_function(log_params):
            lambda_derivative_penalty = 0.5 
            if np.any(log_params > 300): return 1e20
            try:
                linear_params = 10**log_params
                if np.any(np.isinf(linear_params)): return 1e20
                generated_response = fn(linear_params, f)
                value_mse = np.mean((generated_response - err)**2)
                derivative_mse = np.mean((np.diff(generated_response) - np.diff(err))**2)
                total_loss = value_mse + lambda_derivative_penalty * derivative_mse
                if not np.isfinite(total_loss): return 1e20
                return total_loss
            except (ValueError, FloatingPointError): return 1e20
        initial_sigma = 0.5
        log_res, _ = cma.fmin2(log_objective_function, log_p_initial, initial_sigma, options={'verbose': -9, 'maxfevals': 2000})
        return 10**log_res

class Optimizer_WeightedLoss:
    """The final optimizer that uses a template-based weighted loss."""
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected:
            p_initial = [1000.0] * num_params_expected
        template_response = fn(p_initial, f)
        weighting_mask = np.abs(np.diff(template_response))
        if np.sum(weighting_mask) > 0:
            weighting_mask = weighting_mask / np.sum(weighting_mask)
        weighting_mask = np.append(weighting_mask, weighting_mask[-1])
        log_p_initial = np.log10(p_initial)
        def log_objective_function(log_params):
            if np.any(log_params > 300): return 1e20
            try:
                linear_params = 10**log_params
                if np.any(np.isinf(linear_params)): return 1e20
                generated_response = fn(linear_params, f)
                squared_errors = (generated_response - err)**2
                weighted_mse = np.mean(squared_errors * weighting_mask)
                if not np.isfinite(weighted_mse): return 1e20
                return weighted_mse
            except (ValueError, FloatingPointError): return 1e20
        initial_sigma = 0.5
        log_res, _ = cma.fmin2(log_objective_function, log_p_initial, initial_sigma, options={'verbose': -9, 'maxfevals': 2000})
        return 10**log_res

# --- Helper Functions ---

def load_trained_agent(model_path, num_freq_points, num_actions):
    """Loads a trained agent's weights and normalization statistics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    agent = RLAgent(num_freq_points=num_freq_points, num_actions=num_actions)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.state_mean = checkpoint['state_mean']
    agent.state_std = checkpoint['state_std']
    agent.actor.eval()
    agent.critic.eval()
    return agent

def run_inference(agent, optimizer, target_response, freqs, num_stages):
    """Runs the inference process with a specific agent and optimizer."""
    cascaded_response = np.zeros_like(freqs)
    current_error = target_response - cascaded_response
    for _ in range(num_stages):
        state_array = np.array([current_error, target_response, cascaded_response])
        action_index = agent.select_subcircuit(state_array, deterministic=True)
        
        # --- FIX: Access SUBCIRCUITS directly, as it's now imported ---
        subcircuit_info = SUBCIRCUITS[action_index]
        
        best_params = optimizer.optimize(
            subcircuit_info["func"], subcircuit_info["initial_params"], 
            current_error, freqs, subcircuit_info["num_params"]
        )
        optimized_response = subcircuit_info["func"](best_params, freqs)
        cascaded_response += optimized_response
        current_error = target_response - cascaded_response
    return cascaded_response

def generate_benchmark_set(num_responses, max_stages, freqs, w_c_range, zeta_range, available_filters):
    """Generates a diverse and repeatable set of test cases."""
    print(f"Generating a benchmark set of {num_responses} responses...")
    benchmark_set = []
    num_per_stage = num_responses // max_stages
    for stage_count in range(1, max_stages + 1):
        for _ in range(num_per_stage):
            response, _ = circuit_utils.generate_random_circuit_response(
                num_stages=stage_count, freq=freqs, w_c_range=w_c_range, 
                zeta_range=zeta_range, available_filters=available_filters
            )
            benchmark_set.append(response)
    random.shuffle(benchmark_set)
    return benchmark_set

def plot_comparison(results_a, results_b, name_a, name_b):
    """Creates a scatter plot comparing the MSE performance of two models."""
    plt.figure(figsize=(10, 10))
    max_val = max(max(results_a), max(results_b)) * 1.2
    min_val = min(min(results_a), min(results_b)) * 0.8
    plt.scatter(results_a, results_b, alpha=0.6, label='Individual Test Cases')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (Equal Performance)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(f"{name_a} - Final MSE (Lower is Better)")
    plt.ylabel(f"{name_b} - Final MSE (Lower is Better)")
    plt.title("Head-to-Head Model Performance Comparison")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.axis('square')
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_DERIVATIVE_PATH = 'trained_models/agent_opt3_linc_ep_5000.pth' # <-- SET PATH TO OLDER MODEL
    MODEL_WEIGHTED_PATH = 'trained_models/opt5_deriv_att_10000.pth'      # <-- SET PATH TO CURRENT MODEL
    
    BENCHMARK_SIZE = 100
    MAX_GT_STAGES = 5
    NUM_INFERENCE_STAGES = 5

    # --- Setup ---
    freqs = np.logspace(1, 5, 125)
    omega_c_range = circuit_utils.DEFAULT_OMEGA_C_RANGE
    zeta_range = circuit_utils.DEFAULT_ZETA_RANGE
    
    # --- Instantiate Correct Optimizers ---
    optimizer_derivative = Optimizer_DerivativeAware()
    optimizer_weighted = Optimizer_WeightedLoss()
    
    # --- Load Models ---
    print("--- Loading Models ---")
    agent_derivative = load_trained_agent(MODEL_DERIVATIVE_PATH, len(freqs), len(SUBCIRCUITS))
    agent_weighted = load_trained_agent(MODEL_WEIGHTED_PATH, len(freqs), len(SUBCIRCUITS))
    
    # --- Generate Benchmark ---
    benchmark_data = generate_benchmark_set(
        num_responses=BENCHMARK_SIZE, max_stages=MAX_GT_STAGES, freqs=freqs,
        w_c_range=omega_c_range, zeta_range=zeta_range, available_filters=circuit_utils.FILTER_FUNCTIONS
    )
    
    # --- Run Benchmark ---
    print(f"\n--- Running Benchmark on {len(benchmark_data)} Test Cases ---")
    results_derivative = []
    results_weighted = []
    
    wins_derivative = 0; wins_weighted = 0; ties = 0

    for target in tqdm(benchmark_data, desc="Comparing Models"):
        # Run Derivative Model
        response_deriv = run_inference(agent_derivative, optimizer_derivative, target, freqs, NUM_INFERENCE_STAGES)
        mse_deriv = np.mean((target - response_deriv)**2)
        results_derivative.append(mse_deriv)
        
        # Run Weighted Model
        response_weight = run_inference(agent_weighted, optimizer_weighted, target, freqs, NUM_INFERENCE_STAGES)
        mse_weight = np.mean((target - response_weight)**2)
        results_weighted.append(mse_weight)
        
        # Compare results
        if mse_weight < mse_deriv * 0.99: wins_weighted += 1
        elif mse_deriv < mse_weight * 0.99: wins_derivative += 1
        else: ties += 1

    # --- Report Results ---
    avg_mse_deriv = np.mean(results_derivative)
    avg_mse_weight = np.mean(results_weighted)
    
    print("\n--- BENCHMARK COMPLETE ---")
    print(f"\nModel A (Derivative-Aware Optimizer):")
    print(f"  - Average Final MSE: {avg_mse_deriv:.4e}")
    
    print(f"\nModel B (Weighted Loss Optimizer):")
    print(f"  - Average Final MSE: {avg_mse_weight:.4e}")
    
    print("\n--- Head-to-Head Results ---")
    print(f"  - Model A (Derivative) Wins: {wins_derivative}")
    print(f"  - Model B (Weighted)   Wins: {wins_weighted}")
    print(f"  - Ties:                      {ties}")
    
    # --- Visualize ---
    plot_comparison(results_derivative, results_weighted, "Model A (Derivative)", "Model B (Weighted)")