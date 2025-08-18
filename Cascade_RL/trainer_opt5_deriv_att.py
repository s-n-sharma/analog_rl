import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma
from collections import namedtuple
import random
import matplotlib.pyplot as plt
import time
import os

import circuit_utils

# --- 1. Neural Network Architectures ---
class PolicyNetwork(nn.Module):
    def __init__(self, num_freq_points, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 5, padding='same') # 3 input channels
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding='same')
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        flattened_size = (num_freq_points // 8) * 128
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.pool1(F.relu(self.conv1(x))))
        x = F.relu(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.pool3(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return F.softmax(action_logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, num_freq_points):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 5, padding='same') # 3 input channels
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding='same')
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        flattened_size = (num_freq_points // 8) * 128
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.pool1(F.relu(self.conv1(x))))
        x = F.relu(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.pool3(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# --- 2. The RLAgent ---
Experience = namedtuple('Experience', ['state', 'action', 'log_prob', 'reward', 'done', 'old_value'])

class RLAgent:
    def __init__(self, num_freq_points, num_actions, lr=1e-4, gamma=0.98, eps_clip=0.2, entropy_coef=0.01, ppo_epochs=15, mini_batch_size=64):
        self.gamma, self.eps_clip, self.entropy_coef = gamma, eps_clip, entropy_coef
        self.ppo_epochs, self.mini_batch_size = ppo_epochs, mini_batch_size
        self.actor = PolicyNetwork(num_freq_points, num_actions)
        self.critic = CriticNetwork(num_freq_points)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.memory, self.critic_loss_history = [], []
        self.state_mean = torch.zeros(3, num_freq_points, dtype=torch.float32)
        self.state_std = torch.ones(3, num_freq_points, dtype=torch.float32)
        self.state_norm_epsilon = 1e-8
        self.actor.train(); self.critic.train()

    def _update_normalization_stats(self, states_batch):
        batch_mean = states_batch.mean(dim=0); batch_std = states_batch.std(dim=0)
        momentum = 0.99
        self.state_mean = momentum * self.state_mean + (1 - momentum) * batch_mean
        self.state_std = momentum * self.state_std + (1 - momentum) * batch_std
        self.state_std = torch.max(self.state_std, torch.tensor(self.state_norm_epsilon))

    def select_subcircuit(self, state_array, deterministic=False):
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        normalized_state_tensor = (state_tensor - self.state_mean) / (self.state_std + self.state_norm_epsilon)
        with torch.no_grad():
            self.actor.eval()
            action_probs = self.actor(normalized_state_tensor)
        if deterministic:
            return torch.argmax(action_probs, dim=1).item()
        else:
            self.critic.eval()
            value = self.critic(normalized_state_tensor).item()
            self.critic.train()
            dist = torch.distributions.Categorical(action_probs)
            action_sample = dist.sample()
            log_prob = dist.log_prob(action_sample)
            return action_sample.item(), log_prob.item(), value

    def update(self):
        if not self.memory: return
        gae_lambda = 0.95
        rewards = torch.tensor([e.reward for e in self.memory], dtype=torch.float32)
        old_values = torch.tensor([e.old_value for e in self.memory], dtype=torch.float32)
        dones = torch.tensor([e.done for e in self.memory], dtype=torch.float32)
        old_log_probs = torch.tensor([e.log_prob for e in self.memory], dtype=torch.float32)
        states = torch.tensor(np.array([e.state for e in self.memory]), dtype=torch.float32)
        actions = torch.tensor([e.action for e in self.memory], dtype=torch.int64)
        
        self._update_normalization_stats(states)
        normalized_states = (states - self.state_mean) / (self.state_std + self.state_norm_epsilon)

        advantages = torch.zeros_like(rewards); last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_values = old_values[t+1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - old_values[t]
            last_gae_lam = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(normalized_states))
            for start_idx in range(0, len(normalized_states), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size; batch_indices = indices[start_idx:end_idx]
                states_batch, actions_batch, old_log_probs_batch = normalized_states[batch_indices], actions[batch_indices], old_log_probs[batch_indices]
                returns_batch, old_values_batch, advantages_batch = returns[batch_indices], old_values[batch_indices], advantages[batch_indices]
                
                state_values_batch = self.critic(states_batch).squeeze()
                new_probs_batch = self.actor(states_batch); new_dist_batch = torch.distributions.Categorical(new_probs_batch)
                new_log_probs_batch = new_dist_batch.log_prob(actions_batch); entropy_batch = new_dist_batch.entropy().mean()
                
                ratio_batch = torch.exp(new_log_probs_batch - old_log_probs_batch)
                surr1 = ratio_batch * advantages_batch
                surr2 = torch.clamp(ratio_batch, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                loss_unclipped = F.mse_loss(state_values_batch, returns_batch, reduction='none')
                value_clipped = old_values_batch + torch.clamp(state_values_batch - old_values_batch, -self.eps_clip, self.eps_clip)
                loss_clipped = F.mse_loss(value_clipped, returns_batch, reduction='none')
                critic_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
                
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_batch
                self.optimizer.zero_grad(); total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.scheduler.step(); self.memory = []; self.critic_loss_history.append(critic_loss.item())

# --- 3. Optimizer Class ---
class Optimizer:
    def optimize(self, fn, p_initial, err, f, num_params_expected):
        if len(p_initial) != num_params_expected:
            p_initial = [1000.0] * num_params_expected

        # Create a "focus mask" based on the chosen filter type
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
                
                generated_response_for_error = fn(linear_params, f)
                
                # Calculate Weighted MSE
                squared_errors = (generated_response_for_error - err)**2
                weighted_squared_errors = squared_errors * weighting_mask
                weighted_mse = np.mean(weighted_squared_errors)

                if not np.isfinite(weighted_mse):
                    return 1e20
                
                return weighted_mse
                
            except (ValueError, FloatingPointError):
                return 1e20

        initial_sigma = 0.5
        log_res, _ = cma.fmin2(
            log_objective_function, 
            log_p_initial, 
            initial_sigma, 
            options={'verbose': -9, 'maxfevals': 2000}
        )
        return 10**log_res

# --- 4. Environment Components ---
SUBCIRCUITS = [{"name": "LP", "func": circuit_utils.lp_filter, "initial_params": [1000], "num_params": 1}, {"name": "HP", "func": circuit_utils.hp_filter, "initial_params": [1000], "num_params": 1}, {"name": "BRCBS", "func": circuit_utils.brcbs_filter, "initial_params": [1000, 1000], "num_params": 2}]

# --- 5. Plotting ---
episode_numbers, total_rewards, moving_avg_rewards = [], [], []
moving_avg_window = 100
plt.ion(); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
line1, = ax1.plot(episode_numbers, moving_avg_rewards, 'b-', label=f'Moving Average Reward ({moving_avg_window} ep)'); ax1.set_title('RL Agent Training Progress'); ax1.set_ylabel('Moving Average Reward'); ax1.grid(True); ax1.legend()
line2, = ax2.plot([], [], 'r-', label='Critic Loss'); ax2.set_xlabel('Episode'); ax2.set_ylabel('Critic Loss'); ax2.grid(True); ax2.legend()
def update_plots(episodes_per_agent_update):
    line1.set_data(episode_numbers, moving_avg_rewards); ax1.relim(); ax1.autoscale_view()
    if len(agent.critic_loss_history) >= moving_avg_window:
        critic_loss_ma = np.convolve(agent.critic_loss_history, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        num_critic_updates = len(agent.critic_loss_history); x_values_for_updates = np.array(range(1, num_critic_updates + 1)) * episodes_per_agent_update
        critic_ma_x_values = x_values_for_updates[moving_avg_window - 1:]
        line2.set_data(critic_ma_x_values, critic_loss_ma); ax2.relim(); ax2.autoscale_view()
    fig.canvas.draw(); fig.canvas.flush_events()

# --- 6. Main Training Loop ---
if __name__ == '__main__':
    # --- Basic Hyperparameters ---
    num_episodes = 50000
    num_stages_per_episode = 3
    episodes_per_agent_update = 20
    
    # --- Environment Setup ---
    freqs = np.logspace(1, 5, 125)
    omega_c_range = circuit_utils.DEFAULT_OMEGA_C_RANGE
    zeta_range = circuit_utils.DEFAULT_ZETA_RANGE
    
    # --- Agent and Optimizer Setup ---
    agent = RLAgent(num_freq_points=len(freqs), num_actions=len(SUBCIRCUITS), lr=1e-4, gamma=0.98, entropy_coef=0.01)
    optimizer = Optimizer()
    
    # --- Advanced Curriculum Learning Setup ---
    current_max_stages = 2
    max_possible_stages = 5
    is_in_transition = False
    transition_progress = 0.0
    transition_duration = 2500
    reward_threshold = 3.0
    level_up_stability = 2000
    performance_met_at = 0

    print(f"--- 🌱 Starting Curriculum Stage 1 (Max Target Stages: {current_max_stages}) ---")

    # --- Reward and Saving Setup ---
    STEP_PENALTY = 0.1
    TERMINAL_REWARD_SCALAR = 100.0
    BRCBS_PENALTY = 1.0
    SAVE_DIR = 'trained_models'
    CHECKPOINT_INTERVAL = 5000
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("--- Starting Training ---")
    for episode in range(num_episodes):
        
        # --- Curriculum-based Episode Generation ---
        num_gen_stages = current_max_stages
        if is_in_transition:
            if random.random() < transition_progress:
                num_gen_stages = current_max_stages + 1
        
        target_response, _ = circuit_utils.generate_random_circuit_response(
            num_stages=random.randint(1, num_gen_stages), 
            freq=freqs, w_c_range=omega_c_range, zeta_range=zeta_range, 
            available_filters=circuit_utils.FILTER_FUNCTIONS
        )
        
        # --- Run Single Episode ---
        cascaded_response = np.zeros_like(freqs)
        total_episode_reward = 0
        current_error = target_response - cascaded_response

        for stage in range(num_stages_per_episode):
            mse_before = np.mean(current_error**2)
            state_array = np.array([current_error, target_response, cascaded_response])
            
            action_index, log_prob, old_value = agent.select_subcircuit(state_array)
            subcircuit_info = SUBCIRCUITS[action_index]
            
            best_params = optimizer.optimize(subcircuit_info["func"], subcircuit_info["initial_params"], current_error, freqs, subcircuit_info["num_params"])
            
            optimized_response = subcircuit_info["func"](best_params, freqs)
            cascaded_response += optimized_response
            next_error = target_response - cascaded_response
            mse_after = np.mean(next_error**2)
            
            improvement_reward = np.log(mse_before + 1e-9) - np.log(mse_after + 1e-9)
            reward = improvement_reward - STEP_PENALTY
            if subcircuit_info['name'] == "BRCBS":
                reward -= BRCBS_PENALTY
            
            done = (stage == num_stages_per_episode - 1)
            if done:
                reward += TERMINAL_REWARD_SCALAR / (1 + mse_after)

            agent.memory.append(Experience(state_array, action_index, log_prob, reward, done, old_value))
            total_episode_reward += reward
            current_error = next_error

        # --- Logging and Curriculum Advancement ---
        total_rewards.append(total_episode_reward)
        episode_numbers.append(episode + 1)
        current_moving_avg = np.mean(total_rewards[-moving_avg_window:]) if len(total_rewards) >= moving_avg_window else np.mean(total_rewards)
        moving_avg_rewards.append(current_moving_avg)

        if is_in_transition:
            transition_progress += 1.0 / transition_duration
            if transition_progress >= 1.0:
                is_in_transition = False
                transition_progress = 0.0
                current_max_stages += 1
                performance_met_at = 0
                print(f"\n--- ✅ CURRICULUM STABILIZED at Stage {current_max_stages} ---\n")
        
        elif current_max_stages < max_possible_stages:
            if current_moving_avg > reward_threshold:
                if performance_met_at == 0: performance_met_at = episode
                if episode - performance_met_at > level_up_stability:
                    is_in_transition = True
                    reward_threshold += 1.0
                    new_gamma = min(0.995, agent.gamma + 0.005)
                    print(f"\n--- 🚀 STARTING TRANSITION to Stage {current_max_stages + 1} | Gamma: {agent.gamma:.3f} -> {new_gamma:.3f} ---\n")
                    agent.gamma = new_gamma
            else:
                performance_met_at = 0
        
        # --- Agent Update, Plotting, and Saving ---
        if (episode + 1) % episodes_per_agent_update == 0:
            agent.update()
            
        if (episode + 1) % 50 == 0:
            current_critic_loss = agent.critic_loss_history[-1] if agent.critic_loss_history else np.nan
            print(f"Episode {episode+1}/{num_episodes} | MA Reward: {current_moving_avg:.2f} | Critic Loss: {current_critic_loss:.4f} | LR: {agent.scheduler.get_last_lr()[0]:.1e}")
            update_plots(episodes_per_agent_update)
            plt.pause(0.01)

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            path = os.path.join(SAVE_DIR, f'opt5_deriv_att_{episode+1}.pth')
            torch.save({
                'episode': episode + 1,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'state_mean': agent.state_mean,
                'state_std': agent.state_std,
                'max_target_stages': current_max_stages
            }, path)
            print(f"--- Model checkpoint saved at {path} ---")

    print("--- Training Finished ---")
    final_path = os.path.join(SAVE_DIR, 'agent_final.pth')
    torch.save({
        'episode': num_episodes,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_mean': agent.state_mean,
        'state_std': agent.state_std,
        'max_target_stages': current_max_stages
    }, final_path)
    print(f"--- Final model saved at {final_path} ---")
    
    plt.ioff()
    plt.show()