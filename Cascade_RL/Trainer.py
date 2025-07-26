# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma
from collections import namedtuple
import random
import matplotlib.pyplot as plt
import time
import os # Import os for path manipulation

# Import the refactored functions and configurations
import circuit_utils

# --- 1. Neural Network Architectures ---

class PolicyNetwork(nn.Module):
    """The Actor: Decides which action to take. Expanded architecture."""
    def __init__(self, num_freq_points, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding='same') # Increased channels
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding='same') # Increased channels
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding='same') # New layer
        self.pool3 = nn.MaxPool1d(2) # New pooling layer
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened_size after all pooling layers
        # num_freq_points // (2*2*2) = num_freq_points // 8
        flattened_size = (num_freq_points // 8) * 128 # Adjusted for 3 pooling layers and 128 channels
        
        self.fc1 = nn.Linear(flattened_size, 256) # Increased neurons
        self.fc2 = nn.Linear(256, 128) # New layer
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension (batch, channels, length)
        x = F.relu(self.pool1(F.relu(self.conv1(x))))
        x = F.relu(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.pool3(F.relu(self.conv3(x)))) # New layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # New layer
        action_logits = self.fc3(x)
        return F.softmax(action_logits, dim=-1)

class CriticNetwork(nn.Module):
    """The Critic: Estimates the value (expected reward) of a state. Expanded architecture."""
    def __init__(self, num_freq_points):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding='same') # Increased channels
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding='same') # Increased channels
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding='same') # New layer
        self.pool3 = nn.MaxPool1d(2) # New pooling layer
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened_size after all pooling layers
        flattened_size = (num_freq_points // 8) * 128 # Adjusted for 3 pooling layers and 128 channels
        
        self.fc1 = nn.Linear(flattened_size, 256) # Increased neurons
        self.fc2 = nn.Linear(256, 128) # New layer
        self.fc3 = nn.Linear(128, 1) # Outputs a single value

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension (batch, channels, length)
        x = F.relu(self.pool1(F.relu(self.conv1(x))))
        x = F.relu(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.pool3(F.relu(self.conv3(x)))) # New layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # New layer
        value = self.fc3(x)
        return value

# --- 2. The RLAgent with Training Logic ---

Experience = namedtuple('Experience', ['state', 'action', 'log_prob', 'reward', 'done'])

class RLAgent:
    def __init__(self, num_freq_points, num_actions, lr=1e-4, gamma=0.95, eps_clip=0.2, entropy_coef=0.005, ppo_epochs=5, mini_batch_size=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.actor = PolicyNetwork(num_freq_points, num_actions)
        self.critic = CriticNetwork(num_freq_points)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.memory = []
        self.critic_loss_history = []

        # For state normalization (running statistics)
        # Initialize mean to zeros, std to ones (or small value)
        self.state_mean = torch.zeros(num_freq_points, dtype=torch.float32)
        self.state_std = torch.ones(num_freq_points, dtype=torch.float32) # Initialize to 1.0 to avoid division by zero
        self.state_norm_epsilon = 1e-8 # Small epsilon for std dev to prevent division by zero

        # Set initial training mode (important for state norm stats update)
        self.actor.train()
        self.critic.train()

    def _update_normalization_stats(self, states_batch):
        # Update running mean and std using a batch of states
        
        # Ensure states_batch is 2D: (batch_size, num_freq_points)
        if states_batch.dim() == 1: # If for some reason a single sample comes here
             states_batch = states_batch.unsqueeze(0)
        
        # Calculate mean and std for the current batch
        batch_mean = states_batch.mean(dim=0)
        # std() of a single sample is problematic, so we ensure we have >1 samples for robust std
        # and use 0.0 if not enough data, then clamp with epsilon.
        batch_std = states_batch.std(dim=0) if states_batch.shape[0] > 1 else torch.zeros_like(batch_mean)
        
        # Momentum update for running mean and std
        momentum = 0.99
        self.state_mean = momentum * self.state_mean + (1 - momentum) * batch_mean
        self.state_std = momentum * self.state_std + (1 - momentum) * batch_std
        
        # Ensure std doesn't go below a certain threshold to prevent division by zero
        self.state_std = torch.max(self.state_std, torch.tensor(self.state_norm_epsilon))


    def select_subcircuit(self, error_curve):
        state_tensor = torch.FloatTensor(error_curve).unsqueeze(0) # Shape: (1, num_freq_points)
        
        # Apply state normalization using CURRENT running stats
        normalized_state_tensor = (state_tensor - self.state_mean) / (self.state_std + self.state_norm_epsilon)

        with torch.no_grad():
            # Temporarily set to eval for inference within training loop
            # This is important if your networks have BatchNorm or Dropout
            self.actor.eval()
            action_probs = self.actor(normalized_state_tensor)
            self.actor.train() # Set back to train mode

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def update(self):
        """Performs the PPO update."""
        if not self.memory:
            return

        # Convert memory to tensors
        rewards = torch.tensor([e.reward for e in self.memory], dtype=torch.float32)
        old_log_probs = torch.tensor([e.log_prob for e in self.memory], dtype=torch.float32)
        states = torch.tensor(np.array([e.state for e in self.memory]), dtype=torch.float32) # Original unnormalized states
        actions = torch.tensor([e.action for e in self.memory], dtype=torch.int64)

        # --- Update Normalization Statistics BEFORE Normalizing the batch ---
        # Update running mean/std using the collected 'states' batch (raw states)
        self._update_normalization_stats(states) 

        # Apply state normalization to the entire batch of collected states for PPO updates
        normalized_states = (states - self.state_mean) / (self.state_std + self.state_norm_epsilon)

        # Calculate discounted returns
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)

        # PPO Optimization Epochs
        for epoch_idx in range(self.ppo_epochs): # Renamed loop variable to avoid conflict
            # Shuffle data and create mini-batches
            indices = torch.randperm(len(normalized_states))
            for start_idx in range(0, len(normalized_states), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_indices = indices[start_idx:end_idx]

                states_batch = normalized_states[batch_indices] # Use normalized states
                actions_batch = actions[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                returns_batch = returns[batch_indices]

                # Get advantages (re-calculate value for this mini-batch)
                state_values_batch = self.critic(states_batch).squeeze()
                advantages_batch = returns_batch - state_values_batch.detach()
                
                # Advantage Normalization
                # Ensure std is not zero before normalizing
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # Get new log probabilities from the current policy
                new_probs_batch = self.actor(states_batch)
                new_dist_batch = torch.distributions.Categorical(new_probs_batch)
                new_log_probs_batch = new_dist_batch.log_prob(actions_batch)

                # Calculate Entropy Bonus
                entropy_batch = new_dist_batch.entropy().mean()

                # Calculate PPO surrogate loss
                ratio_batch = torch.exp(new_log_probs_batch - old_log_probs_batch)
                surr1_batch = ratio_batch * advantages_batch
                surr2_batch = torch.clamp(ratio_batch, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                
                actor_loss_batch = -torch.min(surr1_batch, surr2_batch).mean()
                critic_loss_batch = F.mse_loss(state_values_batch, returns_batch)
                
                total_loss_batch = actor_loss_batch + 0.5 * critic_loss_batch - self.entropy_coef * entropy_batch
                
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.memory = [] # Clear memory after all PPO epochs
        self.critic_loss_history.append(critic_loss_batch.item()) # Store the last critic loss for logging

# --- 3. Environment Components ---

# Define the available subcircuits for the agent to choose from.
SUBCIRCUITS = [
    {"name": "LP", "func": circuit_utils.lp_filter, "initial_params": [1000], "num_params": 1},
    {"name": "HP", "func": circuit_utils.hp_filter, "initial_params": [1000], "num_params": 1},
    {"name": "BRCBS", "func": circuit_utils.brcbs_filter, "initial_params": [1000, 1000], "num_params": 2}
]

class Optimizer:
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected:
            p_initial = [1000.0] * num_params_expected # Re-initialize if length mismatch
        
        obj = lambda params: np.mean((fn(params, f) - err)**2)
        # Add 'maxfevals' to limit optimization time per step
        res, _ = cma.fmin2(obj, p_initial, 0.5, options={'verbose':-9, 'maxfevals': 1000}) 
        return res

# --- 4. The Main Training Loop ---

# Global variables for plotting
episode_numbers = []
total_rewards = []
moving_avg_rewards = []
moving_avg_window = 50 # Window for moving average

# Setup for live plotting
plt.ion() # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # Two subplots

# Plot 1: Moving Average Reward
line1, = ax1.plot(episode_numbers, moving_avg_rewards, 'b-', label='Moving Average Reward')
ax1.set_title('RL Agent Training Progress')
ax1.set_ylabel('Moving Average Reward')
ax1.grid(True)
ax1.legend()

# Plot 2: Critic Loss
line2, = ax2.plot([], [], 'r-', label='Critic Loss')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Critic Loss')
ax2.grid(True)
ax2.legend()

def update_plots(episodes_per_agent_update):
    # Update data in the reward plot
    line1.set_data(episode_numbers, moving_avg_rewards)
    ax1.relim()
    ax1.autoscale_view()
    
    # Update data in the critic loss plot
    if len(agent.critic_loss_history) >= moving_avg_window:
        critic_loss_ma = np.convolve(agent.critic_loss_history, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        
        # Determine the corresponding episode numbers for the moving average critic loss
        num_critic_updates = len(agent.critic_loss_history)
        x_values_for_critic_updates = np.array(range(1, num_critic_updates + 1)) * episodes_per_agent_update
        
        critic_ma_x_values = x_values_for_critic_updates[moving_avg_window - 1:]

        if len(critic_ma_x_values) == len(critic_loss_ma):
            line2.set_data(critic_ma_x_values, critic_loss_ma)
            ax2.relim()
            ax2.autoscale_view()
        else:
            # This should ideally not happen with the corrected logic, but good for debugging
            print(f"Warning: Critic loss plot data length mismatch! X: {len(critic_ma_x_values)}, Y: {len(critic_loss_ma)}")
    else:
        # If not enough data for moving average, clear or don't plot line2
        line2.set_data([], []) # Clear the line if not enough data yet
        ax2.relim()
        ax2.autoscale_view()
    
    # Redraw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- Model Saving Configuration ---
SAVE_DIR = 'trained_models'
CHECKPOINT_INTERVAL = 10000 # Save every 10,000 episodes

if __name__ == '__main__':
    # Hyperparameters
    num_episodes = 50000 # Increased episodes to show checkpointing
    num_stages_per_episode = 3
    episodes_per_agent_update = 20 # Collect experiences from this many episodes before updating agent

    # Setup
    freqs = np.logspace(1, 5, 125)
    omega_c_range = circuit_utils.DEFAULT_OMEGA_C_RANGE
    zeta_range = circuit_utils.DEFAULT_ZETA_RANGE 
    
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Agent initialization with all new hyperparameters
    agent = RLAgent(
        num_freq_points=len(freqs), 
        num_actions=len(SUBCIRCUITS),
        lr=1e-4,          # Reduced learning rate
        gamma=0.95,       # Slightly reduced gamma
        eps_clip=0.2,     # Keep default
        entropy_coef=0.005, # Added entropy coefficient
        ppo_epochs=15,       # PPO optimization epochs per update
        mini_batch_size=64  # Mini-batch size for PPO epochs
    )

    optimizer = Optimizer() # Initialize optimizer

    print("--- Starting Training ---")
    
    for episode in range(num_episodes):
        target_response = circuit_utils.generate_random_circuit_response(
            num_stages=random.randint(1, 5), 
            freq=freqs,
            w_c_range=omega_c_range,
            zeta_range=zeta_range,
            available_filters=circuit_utils.FILTER_FUNCTIONS
        )
        
        cascaded_response = np.zeros_like(freqs)
        total_episode_reward = 0
        
        current_error = target_response - cascaded_response
        initial_mse = np.mean(current_error**2)

        for stage in range(num_stages_per_episode):
            state = current_error
            
            # agent.select_subcircuit handles state normalization internally
            action_index, log_prob = agent.select_subcircuit(state)
            subcircuit_info = SUBCIRCUITS[action_index]
            
            best_params = optimizer.optimize(
                subcircuit_info["func"], 
                subcircuit_info["initial_params"], 
                current_error, 
                freqs,
                subcircuit_info["num_params"]
            )
            
            optimized_response = subcircuit_info["func"](best_params, freqs)
            cascaded_response += optimized_response
            
            next_error = target_response - cascaded_response
            next_mse = np.mean(next_error**2)
            reward = initial_mse - next_mse
            initial_mse = next_mse
            
            total_episode_reward += reward
            done = (stage == num_stages_per_episode - 1)
            
            agent.memory.append(Experience(state, action_index, log_prob, reward, done))
            
            current_error = next_error

        # Store total reward for this episode
        total_rewards.append(total_episode_reward)
        episode_numbers.append(episode + 1)

        # Calculate moving average reward
        if len(total_rewards) >= moving_avg_window:
            current_moving_avg = np.mean(total_rewards[-moving_avg_window:])
        else:
            current_moving_avg = np.mean(total_rewards)
        moving_avg_rewards.append(current_moving_avg)

        # Print current total and moving average reward, and latest critic loss
        # Ensure agent.critic_loss_history is not empty before trying to access its last element
        current_critic_loss = agent.critic_loss_history[-1] if agent.critic_loss_history else np.nan
        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_episode_reward:.4f} | Moving Avg Reward ({moving_avg_window} episodes): {current_moving_avg:.4f} | Critic Loss: {current_critic_loss:.4f}")

        # --- Agent Update Logic (now batched and with PPO epochs) ---
        if (episode + 1) % episodes_per_agent_update == 0:
            agent.update() # Perform update only after 'episodes_per_agent_update' episodes
            
        # Update plots
        if (episode + 1) % 50 == 0: 
            update_plots(episodes_per_agent_update)
            plt.pause(0.01)

        # --- Save model checkpoint ---
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f'trained_rl_agent_checkpoint_episode_{episode+1}.pth')
            torch.save({
                'episode': episode + 1,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'state_mean': agent.state_mean, # Save running state stats
                'state_std': agent.state_std,
            }, checkpoint_path)
            print(f"--- Model checkpoint saved at {checkpoint_path} ---")

    print("--- Training Finished ---")
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, 'trained_rl_agent_final.pth')
    torch.save({
        'episode': num_episodes,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_mean': agent.state_mean,
        'state_std': agent.state_std,
    }, final_model_path)
    print(f"--- Final model saved at {final_model_path} ---")

    plt.ioff() 
    plt.show()