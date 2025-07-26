import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import cma
from Generator import Generator
from collections import namedtuple
from Filter import Filter

# --- 1. The Data Generator ---

class Generator_Wrapper:
    """Creates achievable target curves by simulating circuits."""
        
    def __init__(self, MAX_FILTERS=5, omega_c_range=None, zeta_range=None, stageDecider=None):
        """Initialize the generator with default parameters."""
        self.generator = Generator(MAX_FILTERS, omega_c_range, zeta_range, stageDecider)

    def getCircuit(self, freq=np.logspace(1e2, 1e6, 125)):
        """Returns a new, achievable target response."""
        return torch.from_numpy(self.generator.getCircuit(np.logspace(1e2, 1e6, 125)))

class Optimizer:
    """Finds the best parameters for a chosen subcircuit."""
    def optimize(self, subcircuit_func, initial_params, target_error, freqs):
        
        def objective(params):
            response = subcircuit_func(params, freqs)
            error = np.mean((response - target_error)**2)
            return error

        print(f"  🔍 Optimizing parameters for {subcircuit_func.__name__}...")
        # Use a real optimizer like CMA-ES
        best_params, es = cma.fmin2(objective, initial_params, 0.5, 
                                     options={'maxiter': 50, 'verbose': -9})
        return best_params

Experience = namedtuple('Experience', ['state', 'action', 'log_prob', 'reward', 'done'])


class CriticNetwork(nn.Module):
    """The Critic: Estimates the value (expected reward) of a state."""
    def __init__(self, num_freq_points):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding='same')
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        flattened_size = (num_freq_points // 4) * 32
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 1) # Outputs a single value

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool1(F.relu(self.conv1(x))))
        x = F.relu(self.pool2(F.relu(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class PolicyNetwork(nn.Module):
    """
    A 1D CNN to decide which subcircuit to choose based on an error curve.
    """
    def __init__(self, num_freq_points, num_actions):
        """
        Args:
            num_freq_points (int): The number of points in the Bode plot (e.g., 200).
            num_actions (int): The number of available subcircuits to choose from.
        """
        super(PolicyNetwork, self).__init__()

        # --- Convolutional Layers to extract features from the curve ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # --- Flatten the features to feed into dense layers ---
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened features after conv/pooling
        # Input length / 2 (pool1) / 2 (pool2) * num_channels (conv2)
        flattened_size = (num_freq_points // 4) * 32

        # --- Dense layers to interpret features and decide action ---
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): The input error curve, shape (batch_size, num_freq_points).
        
        Returns:
            A probability distribution over actions.
        """
        # Add a channel dimension for Conv1D: (batch_size, 1, num_freq_points)
        x = x.unsqueeze(1)
        
        # Pass through convolutional blocks
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax to get a probability distribution over actions
        action_probs = F.softmax(x, dim=1)
        
        return action_probs



class RLAgent:
    def __init__(self, num_freq_points, num_actions, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.actor = PolicyNetwork(num_freq_points, num_actions)
        self.critic = CriticNetwork(num_freq_points)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.memory = []

    def select_subcircuit(self, error_curve):
        state_tensor = torch.FloatTensor(error_curve).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
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
        states = torch.tensor(np.array([e.state for e in self.memory]), dtype=torch.float32)
        actions = torch.tensor([e.action for e in self.memory], dtype=torch.int64)

        # Calculate discounted returns
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)

        # Get advantages
        state_values = self.critic(states).squeeze()
        advantages = returns - state_values.detach()
        
        # Get new log probabilities from the current policy
        new_probs = self.actor(states)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        # Calculate PPO surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(state_values, returns)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory = [] # Clear memory after update

# --- 3. The Main Autoregressive Loop ---

SUBCIRCUITS = {
    "low_pass": {
        "func": Filter().LP,
        "params": 2,  # w_c, zeta
    },
    "high_pass": {
        "func": Filter().HP,
        "params": 2,  # w_c, zeta
    },
    "band_stop": { 
        "func": Filter().BRCBS,
        "params": 4,  # w_c1, zeta_1, w_c2, zeta_2
    }
}

if __name__ == '__main__':
    # Hyperparameters
    num_episodes = 500
    num_stages_per_episode = 3
    
    # Setup
    freqs = np.logspace(1e2, 1e6, 125)  # Frequency range from 100 Hz to 1 MHz
    generator = Generator_Wrapper()
    agent = RLAgent(num_freq_points=len(freqs), num_actions=len(SUBCIRCUITS))
    optimizer = Optimizer()
    
    print("--- Starting Training ---")
    for episode in range(num_episodes):
        target_response = generator.getCircuit(freqs)
        cascaded_response = np.zeros_like(freqs)
        total_episode_reward = 0
        
        current_error = target_response - cascaded_response
        initial_mse = np.mean(current_error**2)

        # Autoregressive loop for one episode
        for stage in range(num_stages_per_episode):
            state = current_error
            
            # Agent selects an action and gets its log probability
            action_index, log_prob = agent.select_subcircuit(state)
            subcircuit = SUBCIRCUITS[action_index]
            
            # Optimizer finds best parameters
            best_params = optimizer.optimize(subcircuit["func"], [1000], current_error, freqs)
            
            # Apply the optimized filter
            optimized_response = subcircuit["func"](best_params, freqs)
            cascaded_response += optimized_response
            
            # Calculate reward as the reduction in MSE
            next_error = target_response - cascaded_response
            next_mse = np.mean(next_error**2)
            reward = initial_mse - next_mse  # Reward is how much we improved
            initial_mse = next_mse # Update the baseline error
            
            total_episode_reward += reward
            done = (stage == num_stages_per_episode - 1)
            
            # Store the experience
            agent.memory.append(Experience(state, action_index, log_prob, reward, done))
            
            current_error = next_error

        # Update the agent's networks after the episode
        agent.update()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_episode_reward:.4f}")

    print("--- Training Finished ---")