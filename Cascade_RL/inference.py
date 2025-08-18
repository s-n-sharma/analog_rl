import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Import the classes and utilities from your training script
from trainer import PolicyNetwork, CriticNetwork, RLAgent, Optimizer, SUBCIRCUITS
import circuit_utils

def load_trained_agent(model_path, num_freq_points, num_actions):
    """Loads the trained agent and its normalization statistics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    agent = RLAgent(num_freq_points=num_freq_points, num_actions=num_actions)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.state_mean = checkpoint['state_mean']
    agent.state_std = checkpoint['state_std']
    
    print(f"--- Model loaded successfully from {model_path} ---")
    agent.actor.eval()
    agent.critic.eval()
    return agent

def run_inference(agent, target_response, freqs, num_stages=3):
    """Runs the inference process to build a filter for the target response."""
    optimizer = Optimizer()
    cascaded_response = np.zeros_like(freqs)
    current_error = target_response - cascaded_response
    chosen_circuits = []
    
    print("\n--- Starting Inference ---")
    for stage in range(num_stages):
        state_array = np.array([current_error, target_response, cascaded_response])
        action_index = agent.select_subcircuit(state_array, deterministic=True)
        subcircuit_info = SUBCIRCUITS[action_index]
        
        print(f"Stage {stage + 1}: Agent chose '{subcircuit_info['name']}' topology.")
        
        best_params = optimizer.optimize(
            subcircuit_info["func"], 
            subcircuit_info["initial_params"], 
            current_error, 
            freqs,
            subcircuit_info["num_params"]
        )
        
        optimized_response = subcircuit_info["func"](best_params, freqs)
        cascaded_response += optimized_response
        current_error = target_response - cascaded_response
        chosen_circuits.append({'name': subcircuit_info['name'], 'params': best_params, 'response': optimized_response})

    print("--- Inference Finished ---")
    return cascaded_response, chosen_circuits

def plot_results(target, final_response, chosen_circuits, freqs, final_loss):
    """Plots the target, the final result, and displays the final loss."""
    plt.figure(figsize=(12, 8))
    
    plt.semilogx(freqs, target, 'k--', linewidth=2.5, label='Target Response')
    plt.semilogx(freqs, final_response, 'b-', linewidth=2.5, label='Final Cascaded Response')
    
    for i, circuit in enumerate(chosen_circuits):
        plt.semilogx(freqs, circuit['response'], '--', label=f"Stage {i+1}: {circuit['name']}")

    plt.title(f'Inference Result\nFinal MSE Loss: {final_loss:.4e}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    # --- FIX: Ensure the plot window stays open ---
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = 'trained_models/agent_opt3_linc_ep_10000.pth'
    NUM_STAGES = 10
    
    # --- Setup ---
    freqs = np.logspace(1, 5, 125)
    omega_c_range = circuit_utils.DEFAULT_OMEGA_C_RANGE
    zeta_range = circuit_utils.DEFAULT_ZETA_RANGE 
    
    # --- Load Agent ---
    agent = load_trained_agent(
        model_path=MODEL_PATH,
        num_freq_points=len(freqs),
        num_actions=len(SUBCIRCUITS)
    )
    
    # --- Generate a Target Function and get Ground Truth ---
    target_response, ground_truth_info = circuit_utils.generate_random_circuit_response(
        num_stages=10, 
        freq=freqs,
        w_c_range=omega_c_range,
        zeta_range=zeta_range,
        available_filters=circuit_utils.FILTER_FUNCTIONS
    )
    
    # --- NEW: Print the ground truth ---
    print("\n--- Ground Truth Circuit ---")
    for i, circuit in enumerate(ground_truth_info):
        param_str = ", ".join([f"{p:.2f}" for p in circuit['params']])
        print(f"  {i+1}. Type: {circuit['name']}, Params: [{param_str}]")
    
    # --- Run Inference ---
    final_cascaded_response, chosen_circuits_info = run_inference(
        agent=agent,
        target_response=target_response,
        freqs=freqs,
        num_stages=NUM_STAGES
    )
    
    # --- NEW: Calculate the final loss ---
    final_loss = np.mean((target_response - final_cascaded_response)**2)
    
    # --- Display Results ---
    print("\n--- Final Agent-Built Circuit ---")
    for i, circuit in enumerate(chosen_circuits_info):
        param_str = ", ".join([f"{p:.2f}" for p in circuit['params']])
        print(f"  {i+1}. Type: {circuit['name']}, Optimized Params: [{param_str}]")
    
    print(f"\nFinal Mean Squared Error: {final_loss:.4e}")

    # --- NEW: Pass the final loss to the plotting function ---
    plot_results(target_response, final_cascaded_response, chosen_circuits_info, freqs, final_loss)