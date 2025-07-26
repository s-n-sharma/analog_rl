# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Import the network architectures (must be identical to training script)
from main_training_script import PolicyNetwork, CriticNetwork, SUBCIRCUITS # Assuming you keep them in main_training_script
from main_training_script import Optimizer # Need Optimizer for inference too
import circuit_utils # For filter functions and target generation

# --- Configuration ---
MODEL_PATH = 'trained_rl_agent_checkpoint.pth' # Path to your saved model
NUM_FREQ_POINTS = 200 # Must match the training setup
NUM_ACTIONS = len(SUBCIRCUITS) # Must match the training setup
FREQ_RANGE = np.logspace(1, 5, NUM_FREQ_POINTS) # Frequencies to evaluate on

# --- Helper function to load the model ---
def load_agent(model_path, num_freq_points, num_actions):
    actor = PolicyNetwork(num_freq_points, num_actions)
    critic = CriticNetwork(num_freq_points)

    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    actor.eval() # Set to evaluation mode
    critic.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {model_path}")
    return actor, critic

# --- Inference Function ---
def run_inference(actor, critic, freqs, num_stages_for_agent=3):
    # Generate a new random target circuit
    target_response = circuit_utils.generate_random_circuit_response(
        num_stages=random.randint(1, 5), # Target can have random stages
        freq=freqs,
        w_c_range=circuit_utils.DEFAULT_OMEGA_C_RANGE,
        zeta_range=circuit_utils.DEFAULT_ZETA_RANGE,
        available_filters=circuit_utils.FILTER_FUNCTIONS
    )

    cascaded_response = np.zeros_like(freqs)
    current_error = target_response - cascaded_response
    optimizer = Optimizer() # Instantiate optimizer for inference

    print("\n--- Running Inference ---")
    print(f"Target generated. Num stages for agent: {num_stages_for_agent}")

    selected_subcircuits = []
    
    # Agent's autoregressive loop
    for stage in range(num_stages_for_agent):
        state_tensor = torch.FloatTensor(current_error).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = actor(state_tensor)
        
        # We'll take the action with the highest probability for inference
        action_index = torch.argmax(action_probs, dim=-1).item()
        
        subcircuit_info = SUBCIRCUITS[action_index]
        selected_subcircuits.append(subcircuit_info["name"])

        print(f"Stage {stage+1}: Agent selected '{subcircuit_info['name']}' (Action Index: {action_index})")
        
        # Optimizer finds best parameters for this selected filter
        best_params = optimizer.optimize(
            subcircuit_info["func"],
            subcircuit_info["initial_params"],
            current_error,
            freqs,
            subcircuit_info["num_params"]
        )
        
        # Apply the optimized filter
        optimized_response = subcircuit_info["func"](best_params, freqs)
        cascaded_response += optimized_response
        
        # Update error for the next stage
        current_error = target_response - cascaded_response

    final_mse = np.mean(current_error**2)
    print(f"Inference complete. Final MSE: {final_mse:.4f}")

    return target_response, cascaded_response, selected_subcircuits, final_mse

# --- Main execution ---
if __name__ == '__main__':
    # Load the trained agent
    actor, critic = load_agent(MODEL_PATH, NUM_FREQ_POINTS, NUM_ACTIONS)

    # Run inference
    target_resp, model_resp, selected_circuits, final_mse = run_inference(actor, critic, FREQ_RANGE)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(FREQ_RANGE, target_resp, label='Target Response', color='blue', linewidth=2)
    plt.plot(FREQ_RANGE, model_resp, label='Model\'s Best Attempt', color='red', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.title(f'Target vs. Model\'s Attempt (Final MSE: {final_mse:.4f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

    print("\nSelected Subcircuits by Agent:")
    for i, circuit_name in enumerate(selected_circuits):
        print(f"  Stage {i+1}: {circuit_name}")