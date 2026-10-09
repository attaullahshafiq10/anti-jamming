import numpy as np
from .reward import calculate_reward

class WirelessEnvironment:
    def __init__(self, data_samples, reward_weights):
        self.data = data_samples
        self.num_samples = len(data_samples)
        self.current_step = 0
        self.reward_weights = reward_weights
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
        
    def _get_state(self):
        return self.data[self.current_step]
        
    def step(self, action):
        # Simulation Logic
        energy_cost = 0.0
        delta_pdr = 0.0
        delta_sinr = 0.0
        delta_throughput = 0.0
        packet_loss = 0.0
        
        if action == 0:
            energy_cost = 1.0
        elif action == 1:
            energy_cost = 5.0
            delta_sinr = 2.0
            delta_pdr = 0.1
        elif action == 2:
            energy_cost = 10.0
            delta_sinr = 5.0
            delta_pdr = 0.3
            delta_throughput = 10.0
            
        reward = calculate_reward(
            delta_pdr, delta_sinr, delta_throughput, packet_loss, energy_cost,
            self.reward_weights['alpha'], self.reward_weights['beta'], 
            self.reward_weights['delta'], self.reward_weights['lambda_'], 
            self.reward_weights['mu']
        )
        
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1
        next_state = self._get_state() if not done else np.zeros_like(self.data[0])
        
        return next_state, reward, done
