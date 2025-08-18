import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

# Import the classes and utilities from your training script
from trainer import RLAgent, Optimizer, SUBCIRCUITS
import circuit_utils

# --- Re-usable Helper Functions (from inference.py) ---

def load_trained_agent(model_path, num_freq_points, num_actions):
    """Loads a trained agent and its normalization statistics."""
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

def run_inference(agent, target_response, freqs, num_stages):
    """Runs the inference process for a single target response."""
    optimizer = Optimizer()
    cascaded_response = np.zeros_like(freqs)
    current_error = target_response - cascaded_response
    
    for _ in range(num_stages):
        state_array = np.array([current_error, target_response, cascaded_response])
        action_index = agent.select_subcircuit(state_array, deterministic=True)
        subcircuit_info = SUBCIRCUITS[action_index]
        
        best_params = optimizer.optimize(
            subcircuit_info["func"], subcircuit_info["initial_params"], 
            current_error, freqs, subcircuit_info["num_params"]
        )
        
        optimized_response = subcircuit_info["func"](best_params, freqs)
        cascaded_response += optimized_response
        current_error = target_response - cascaded_response
        
    return cascaded_response

# --- New Benchmark-Specific Functions ---

def generate_benchmark_set(num_responses, max_stages, freqs, w_c_range, zeta_range, available_filters):
    """Generates a diverse and repeatable set of test cases."""
    print(f"Generating a benchmark set of {num_responses} responses...")
    benchmark_set = []
    num_per_stage = num_responses // max_stages
    
    for stage_count in range(1, max_stages + 1):
        for _ in range(num_per_stage):
            response, ground_truth = circuit_utils.generate_random_circuit_response(
                num_stages=stage_count, freq=freqs, w_c_range=w_c_range, 
                zeta_range=zeta_range, available_filters=available_filters
            )
            benchmark_set.append({'target': response, 'truth': ground_truth})
            
    random.shuffle(benchmark_set) # Shuffle to mix difficulties
    return benchmark_set

def plot_comparison(results_a, results_b):
    """Creates a scatter plot comparing the MSE performance of two models."""
    plt.figure(figsize=(10, 10))
    
    max_val = max(max(results_a), max(results_b))
    min_val = min(min(results_a), min(results_b))
    
    plt.scatter(results_a, results_b, alpha=0.6, label='Individual Test Cases')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (Equal Performance)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Model A - Final MSE (Lower is Better)")
    plt.ylabel(f"Model B - Final MSE (Lower is Better)")
    plt.title("Head-to-Head Model Performance Comparison")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.axis('equal') # Ensure the plot is square for a fair comparison
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_A_PATH = 'trained_models/agent_opt3_linc_ep_5000.pth'  # <-- SET YOUR FIRST MODEL's PATH
    #MODEL_A_PATH = 'trained_models/agent_checkpoint_opt2_ep_10000.pth' # <-- SET YOUR SECOND MODEL's PATH
    #MODEL_B_PATH = 'trained_models/agent_opt3_brcp_ep_5000.pth'  # <-- SET YOUR SECOND MODEL's PATH
    MODEL_B_PATH = 'trained_models/agent_opt3_linc_ep_10000.pth'  # <-- SET YOUR FIRST MODEL's PATH
    
    BENCHMARK_SIZE = 250
    MAX_GT_STAGES = 7  # Generate targets with up to 5 stages
    NUM_INFERENCE_STAGES = 7 # The number of stages our agents get to use

    # --- Setup ---
    freqs = np.logspace(1, 5, 125)
    omega_c_range = circuit_utils.DEFAULT_OMEGA_C_RANGE
    zeta_range = circuit_utils.DEFAULT_ZETA_RANGE
    
    # --- Load Models ---
    print("--- Loading Models ---")
    agent_A = load_trained_agent(MODEL_A_PATH, len(freqs), len(SUBCIRCUITS))
    agent_B = load_trained_agent(MODEL_B_PATH, len(freqs), len(SUBCIRCUITS))
    
    # --- Generate Benchmark ---
    benchmark_data = generate_benchmark_set(
        num_responses=BENCHMARK_SIZE, max_stages=MAX_GT_STAGES, freqs=freqs,
        w_c_range=omega_c_range, zeta_range=zeta_range, available_filters=circuit_utils.FILTER_FUNCTIONS
    )
    
    # --- Run Benchmark ---
    print(f"\n--- Running Benchmark on {len(benchmark_data)} Test Cases ---")
    results_A = []
    results_B = []
    
    wins_A = 0
    wins_B = 0
    ties = 0

    for test_case in tqdm(benchmark_data, desc="Comparing Models"):
        target = test_case['target']
        
        # Run Model A
        response_A = run_inference(agent_A, target, freqs, NUM_INFERENCE_STAGES)
        mse_A = np.mean((target - response_A)**2)
        results_A.append(mse_A)
        
        # Run Model B
        response_B = run_inference(agent_B, target, freqs, NUM_INFERENCE_STAGES)
        mse_B = np.mean((target - response_B)**2)
        results_B.append(mse_B)
        
        # Compare results
        if mse_A < mse_B * 0.95: # Model A wins if it's at least 1% better
            wins_A += 1
        elif mse_B < mse_A * 0.95: # Model B wins if it's at least 1% better
            wins_B += 1
        else:
            ties += 1

    # --- Report Results ---
    avg_mse_A = np.mean(results_A)
    avg_mse_B = np.mean(results_B)
    
    print("\n--- BENCHMARK COMPLETE ---")
    print(f"\nModel A ({os.path.basename(MODEL_A_PATH)}):")
    print(f"  - Average Final MSE: {avg_mse_A:.4e}")
    
    print(f"\nModel B ({os.path.basename(MODEL_B_PATH)}):")
    print(f"  - Average Final MSE: {avg_mse_B:.4e}")
    
    print("\n--- Head-to-Head Results ---")
    print(f"  - Model A Wins: {wins_A}")
    print(f"  - Model B Wins: {wins_B}")
    print(f"  - Ties:         {ties}")
    
    # --- Visualize ---
    plot_comparison(results_A, results_B)