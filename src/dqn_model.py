import torch
import torch.nn as nn

class DuelingDoubleDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDoubleDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
