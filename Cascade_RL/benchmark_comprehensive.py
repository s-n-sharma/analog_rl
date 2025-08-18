import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cma
import random
from tqdm import tqdm

# Import the RLAgent and circuit utilities
from trainer import RLAgent
import circuit_utils

# --- Version-Specific Optimizer Classes ---

class Optimizer_DerivativeAware:
    """Version 2: The optimizer that uses a global derivative-aware loss."""
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected: p_initial = [1000.0] * num_params_expected
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
    """Version 3: The final optimizer that uses a template-based weighted loss."""
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected: p_initial = [1000.0] * num_params_expected
        template_response = fn(p_initial, f)
        weighting_mask = np.abs(np.diff(template_response))
        if np.sum(weighting_mask) > 0: weighting_mask = weighting_mask / np.sum(weighting_mask)
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

def generate_action_space(action_space_type='simple', freqs=None):
    """Generates the appropriate SUBCIRCUITS list based on the model version."""
    base_filters = [
        {"base_name": "LP", "func": circuit_utils.lp_filter, "num_params": 1},
        {"base_name": "HP", "func": circuit_utils.hp_filter, "num_params": 1},
        {"base_name": "BRCBS", "func": circuit_utils.brcbs_filter, "num_params": 2}
    ]

    if action_space_type == 'simple':
        return [
            {"name": "LP", "base_name": "LP", "func": circuit_utils.lp_filter, "initial_params": [1000], "num_params": 1},
            {"name": "HP", "base_name": "HP", "func": circuit_utils.hp_filter, "initial_params": [1000], "num_params": 1},
            {"name": "BRCBS", "base_name": "BRCBS", "func": circuit_utils.brcbs_filter, "initial_params": [1000, 1000], "num_params": 2}
        ]
    
    elif action_space_type == 'where':
        NUM_BINS = 10
        bin_centers = np.logspace(np.log10(freqs[0]*10), np.log10(freqs[-1]*0.1), NUM_BINS)
        subcircuits = []
        for bf in base_filters:
            if bf["num_params"] == 1:
                for center_freq in bin_centers:
                    subcircuits.append({
                        "name": f"{bf['base_name']} @ {center_freq:.0f}Hz", "base_name": bf['base_name'], 
                        "func": bf['func'], "initial_params": [center_freq], "num_params": 1
                    })
            elif bf["num_params"] == 2:
                for center_freq in bin_centers:
                    subcircuits.append({
                        "name": f"{bf['base_name']} @ {center_freq:.0f}Hz", "base_name": bf['base_name'], 
                        "func": bf['func'], "initial_params": [center_freq * 0.8, center_freq * 1.2], "num_params": 2
                    })
        return subcircuits
    else:
        raise ValueError("Unknown action_space_type")

def load_trained_agent(model_path, num_freq_points, num_actions):
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}. Skipping.")
        return None
    agent = RLAgent(num_freq_points=num_freq_points, num_actions=num_actions)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.state_mean = checkpoint['state_mean']
    agent.state_std = checkpoint['state_std']
    agent.actor.eval(); agent.critic.eval()
    return agent

def run_inference(agent, optimizer, subcircuits, target_response, freqs, num_stages):
    cascaded_response = np.zeros_like(freqs)
    current_error = target_response - cascaded_response
    for _ in range(num_stages):
        state_array = np.array([current_error, target_response, cascaded_response])
        action_index = agent.select_subcircuit(state_array, deterministic=True)
        subcircuit_info = subcircuits[action_index]
        best_params = optimizer.optimize(
            subcircuit_info["func"], subcircuit_info["initial_params"], 
            current_error, freqs, subcircuit_info["num_params"]
        )
        optimized_response = subcircuit_info["func"](best_params, freqs)
        cascaded_response += optimized_response
        current_error = target_response - cascaded_response
    return cascaded_response

def generate_benchmark_set(num_responses, max_stages, freqs, w_c_range, zeta_range, available_filters):
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

def plot_summary_results(results_dict):
    """Creates a bar chart comparing the average MSE of the models."""
    model_names = list(results_dict.keys())
    avg_mses = [np.mean(mses) for mses in results_dict.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, avg_mses)
    plt.ylabel("Average Final MSE (Lower is Better)")
    plt.title("Benchmark Results: Model Performance Comparison")
    plt.yscale('log')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3e}', va='bottom', ha='center')
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATHS = {
        "Derivative-Aware": 'trained_models/agent_opt3_linc_ep_10000.pth',
        "Weighted-Loss": 'trained_models/opt5_deriv_att_10000.pth',
        "Where-Action": 'trained_models/agent_checkpoint_ep_15000.pth'
    }
    
    BENCHMARK_SIZE = 100
    MAX_GT_STAGES = 10
    NUM_INFERENCE_STAGES = 10

    # --- Setup ---
    freqs = np.logspace(1, 5, 125)
    
    # --- Instantiate All Components ---
    optimizers = {
        "Derivative-Aware": Optimizer_DerivativeAware(),
        "Weighted-Loss": Optimizer_WeightedLoss(),
        "Where-Action": Optimizer_WeightedLoss() # The "Where" model uses the weighted optimizer
    }
    
    action_spaces = {
        "Derivative-Aware": generate_action_space('simple'),
        "Weighted-Loss": generate_action_space('simple'),
        "Where-Action": generate_action_space('where', freqs=freqs)
    }
    
    # --- Load All Models ---
    print("--- Loading All Models ---")
    agents = {}
    for name, path in MODEL_PATHS.items():
        subcircuits = action_spaces[name]
        agent = load_trained_agent(path, len(freqs), len(subcircuits))
        if agent:
            agents[name] = agent
    
    # --- Generate Benchmark ---
    benchmark_data = generate_benchmark_set(
        num_responses=BENCHMARK_SIZE, max_stages=MAX_GT_STAGES, freqs=freqs,
        w_c_range=circuit_utils.DEFAULT_OMEGA_C_RANGE, zeta_range=circuit_utils.DEFAULT_ZETA_RANGE, 
        available_filters=circuit_utils.FILTER_FUNCTIONS
    )
    
    # --- Run Benchmark ---
    print(f"\n--- Running Benchmark on {len(benchmark_data)} Test Cases ---")
    results = {name: [] for name in agents.keys()}

    for target in tqdm(benchmark_data, desc="Benchmarking Models"):
        for name, agent in agents.items():
            optimizer = optimizers[name]
            subcircuits = action_spaces[name]
            
            response = run_inference(agent, optimizer, subcircuits, target, freqs, NUM_INFERENCE_STAGES)
            mse = np.mean((target - response)**2)
            results[name].append(mse)

    # --- Report Results ---
    print("\n--- BENCHMARK COMPLETE ---")
    for name, mses in results.items():
        avg_mse = np.mean(mses)
        print(f"\nModel: {name} ({os.path.basename(MODEL_PATHS[name])})")
        print(f"  - Average Final MSE: {avg_mse:.4e}")
        
    # --- Visualize ---
    if results:
        plot_summary_results(results)